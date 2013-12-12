from __future__ import division
import numpy as np
import itertools
import pandas as pd
import operator
import sys
import argparse
import pickle

from mpi4py import MPI

import pycuda.gpuarray as gpu
import pycuda.compiler as nvcc

##### CUDA Kernel #####
f_source='''
__global__ void my_f(int* d_combinations, float* d_res, int* d_ll,int* d_DF,int nf, int height,int df_height) {

  int indi = blockIdx.x * blockDim.x + threadIdx.x;

  if(indi<height){
      float alpha_0 = 0;
      float alpha_1 = 0;
      float S3_0 = 0;
      float S3_1 = 0;
      int counter = 0;
      for (int row = 0; row <= df_height-1;row++){
          for (int col = 0; col <= nf-1;col++){
            if (d_DF[row*nf+col] == d_combinations[indi*nf+col]){
          counter += 1;
            }
          }
          if (counter==nf){
            if(d_ll[row]==0){
                alpha_0 +=  1;
                S3_0 += log(alpha_0);
            }
            if(d_ll[row]==1){
                alpha_1 +=  1;
                S3_1 += log(alpha_1);
            }
          }
          counter = 0;
     }

      int N = alpha_0 + alpha_1;
      int r = 2;
      float S1 = 0;
      if (N>0){
        for (float n = r; n <= (N+r-1);n++){
          S1 += log(n);
        }
      }
      d_res[indi] = S3_1 +S3_0-S1;
  }
}'''

########## FUNCTIONS ##########


##### Calculating Unique Values of Features in the Dataset #####
def vals_of_attributes(D, n):
    output = []
    for i in xrange(n):
        output.append(list(np.unique(D[:, i])))
    return output


##### Evaluating the f function at initialization #####
def f_initialization(i, attribute_values, df):

    V = attribute_values[i]
    r = len(V)

    #Calculate alpha_0 and alpha_1
    alpha_0 = df[df[i] == 0].shape[0]
    alpha_1 = df[df[i] == 1].shape[0]
    N = alpha_0 + alpha_1

    numerator = np.sum([np.log(b) for b in range(1, r)])
    denominator = np.sum([np.log(b) for b in range(1, r + N)])
    term1 = sum([np.log(x) for x in range(1, alpha_0 + 1)])
    term2 = sum([np.log(x) for x in range(1, alpha_1 + 1)])

    return numerator - denominator + term1 + term2


##### Evaluating the f function for all the combinations in parallel #####
def f_combinations(i, pi, attribute_values, df, f_kernel):

    #Finding the Combinations
    phi_i_ = [attribute_values[item] for item in pi]
    combinations = np.array(list(itertools.product(*phi_i_)), dtype=np.int32)
    combinations = np.array(combinations.ravel().tolist(), dtype=np.int32)

    #Determining BlockSize and Gridsize based on Number of Combinations
    df_height = np.int32(df.shape[0])
    nf = np.int32(df[pi].shape[1])
    height = np.int32(combinations.shape[0])
    block_x = np.int(height / nf)
    blocksize = (block_x, 1, 1)
    gridsize = (1, 1)

    #Apply Kernel
    h_DF = np.array(df[pi], dtype=np.int32)
    h_DF = np.array(h_DF.ravel().tolist(), dtype=np.int32)
    h_ll = np.array(df[i], dtype=np.int32)
    h_res = np.zeros(height / nf, dtype=np.float32)
    d_combinations = gpu.to_gpu(combinations)
    d_res = gpu.to_gpu(h_res)
    d_ll = gpu.to_gpu(h_ll)
    d_DF = gpu.to_gpu(h_DF)
    f_kernel(d_combinations, d_res, d_ll, d_DF, nf, height, df_height,
        block=blocksize, grid=gridsize)

    ress = d_res.get()
    ress = list(ress)

    return sum(ress)


##### Performing the K2 algorithm #####
def k2_in_parallel(D, node_order, comm, rank, size, f_kernel, u=2):
    n = D.shape[1]
    assert len(node_order) == n, ("Node order is not correct length."
        "  It should have length %r" % n)
    attribute_values = vals_of_attributes(D, n)

    df = pd.DataFrame(D)
    OKToProceed = False

    # master node
    if rank == 0:
        parents = {}
        status = MPI.Status()

        for i in range(1, size):
            comm.send(i - 1, dest=i)

        for i in xrange(size - 1, n):
            new_parents = comm.recv(source=MPI.ANY_SOURCE, status=status)
            parents.update(new_parents)
            destin = status.Get_source()
            comm.send(i, dest=destin)

        for i in range(1, size):
            new_parents = comm.recv(source=MPI.ANY_SOURCE, status=status)
            parents.update(new_parents)
            destin = status.Get_source()
            comm.send(None, dest=destin)

        return parents

    # slave nodes
    else:
        while (True):
            i = comm.recv(source=0)

            if i is None:
                return

            else:
                OKToProceed = False
                pi = []
                pred = node_order[0:i]
                P_old = f_initialization(node_order[i], attribute_values, df)
                if len(pred) > 0:
                    OKToProceed = True

                while (OKToProceed is True and len(pi) < u):
                    iters = [item for item in pred if item not in pi]
                    if len(iters) > 0:
                        f_to_max = {}

                        for z_hat in iters:
                            f_to_max[z_hat] = f_combinations(node_order[i],
                                pi + [z_hat], attribute_values, df, f_kernel)

                        z = max(f_to_max.iteritems(),
                            key=operator.itemgetter(1))[0]
                        P_new = f_to_max[z]
                        if P_new > P_old:
                            P_old = P_new
                            pi = pi + [z]
                        else:
                            OKToProceed = False
                    else:
                        OKToProceed = False

                message = {node_order[i]: pi}

                comm.send(message, dest=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''K2 In Parallel:  Calculates
         the parent set for each node in your data file and returns a
         dictionary of the form{feature: [parent set]}.''')
    parser.add_argument('-D', nargs='?', default=None, help='''Path to csc file
         containing a 0/1 array with m observations (rows) and n features
         (columns).  A value of 1 represents the presence of that feature in
         that observation. One of --random and -D must be used.''')
    parser.add_argument('--node_order', '-o', nargs='?', type=list,
        default=None, help='''A list of integers containing the column order
        of features in your matrix.  If not provided, order the features in
        accordance with their order in the file.''')
    parser.add_argument('--random', '-r', action="store_true",
        help='''Include this option to calculate parents for a random matrix.
        If --random is included, -D and --node_order should be left out, and
        -m, --seed, and -n can be included.   One of --random and -D ust be
        used.''')
    parser.add_argument('--seed', nargs='?', type=int, default=None,
            help='The seed for the random matrix.  Only use with --random')
    parser.add_argument('-n', nargs='?', type=int, default='10',
            help='''The number of features in a random matrix.
            Default is 10.  Only use with --random''')
    parser.add_argument('-m', nargs='?', type=int, default='100',
        help='''The number of observations in a random matrix.
        Default is 100. Only use with --random''')
    parser.add_argument('-u', nargs='?', type=int, default=2,
        help='''The maximum number of parents per feature.  Default is 2.
                Must be less than number of features.''')
    parser.add_argument('--outfile', nargs='?', default=None, help='''The
         output file where the dictionary of {feature: [parent set]} will be
         written''')
    args = parser.parse_args()

    ##### Load Kernel and MPI Size and Ranks #####
    f_kernel = nvcc.SourceModule(f_source).get_function("my_f")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ##### Manage Argument Cases #####
    u = args.u
    outfile = args.outfile

    if args.random:
        n = args.n
        m = args.m
        if rank == 0:
            if args.seed is not None:
                np.random.seed(args.seed)
            D = np.random.binomial(1, 0.9, size=(m, n))
        else:
            D = None
        D = comm.bcast(D, root=0)
        node_order = list(range(n))

    elif args.D is not None:
        D = np.loadtxt(open(args.D))
        if args.node_order is not None:
            node_order = args.node_order
        else:
            n = np.int32(D.shape[1])
            node_order = list(range(n))

    else:
        if rank == 0:
            print "Incorrect usage. Use --help to display help."
        sys.exit()

    comm.barrier()
    start = MPI.Wtime()
    parents = k2_in_parallel(D, node_order, comm, rank, size, f_kernel, u=u)
    comm.barrier()
    end = MPI.Wtime()

    ##### Outputs #####
    if rank == 0:
        print "Parallel computing time", end - start

        if args.outfile is not None:
            out = open(outfile, 'w')
            try:
                pickle.dump(parents, out)
            except RuntimeError:
                for key, item in parents.iteritems():
                    strr = str(key) + ' ' + str(item) + '\n'
                    out.write(strr)
        else:
            print parents
