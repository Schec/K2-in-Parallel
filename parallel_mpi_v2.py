from __future__ import division
import numpy as np
import itertools
import pandas as pd
import math
import operator
import time
from mpi4py import MPI

import jodys_serial_v2 as serialv
import parallel_mpi_v1 as oldparallel

def vals_of_attributes(D,n):
    output = []
    for i in xrange(n):
        output.append(list(np.unique(D[:,i])))
    return output

def alpha(df, mask):
    _df = df
    for combo in mask:
        _df = _df[_df[combo[0]] == combo[1]]  # I know there must be a way to speed this up - but i couldn't find it
    return len(_df)

def f(i,pi,attribute_values,df):

    len_pi = len(pi)

    phi_i_ = [attribute_values[item] for item in pi]
    if len(phi_i_) == 1:
        phi_i = [[item] for item in phi_i_[0]]
    else:
        phi_i = list(itertools.product(*phi_i_))

    # bug fix: phi_i might contain empty tuple (), which shouldn't be counted in q_I
    try:
        phi_i.remove(())
    except ValueError:
        pass

    q_i = len(phi_i)

    V_i = attribute_values[i]
    r_i = len(V_i)

    #product = 1
    product = 0
    #numerator = math.factorial(r_i - 1)
    numerator = np.sum([np.log(b) for b in range(1, r_i)])

    # special case: q_i = 0
    if q_i == 0:
        js = ['special']
    else:
        js = range(q_i) 

    for j  in js:

        # initializing mask to send to alpha
        if j == 'special':
            mask = []
        else:
            mask = zip(pi,phi_i[j])

        # initializing counts that will increase with alphas
        N_ij = 0
        #inner_product = 1
        inner_product = 0

        for k in xrange(r_i):
            # adjusting mask for each k
            mask_with_k = mask + [[i,V_i[k]]]
            alpha_ijk = alpha(df,mask_with_k)
            N_ij += alpha_ijk
            #inner_product = inner_product*math.factorial(alpha_ijk)
            inner_product = inner_product + np.sum([np.log(b) for b in range(1, alpha_ijk+1)])
        #denominator = math.factorial(N_ij + r_i - 1)
        denominator = np.sum([np.log(b) for b in range(1, N_ij+r_i)])
        #product = product*(numerator/denominator)*inner_product
        product = product + numerator - denominator + inner_product
    return product

def my_job(i,rank,size):
    flag = False
    if np.floor(i/size) % 2 == 0 and i%size == rank:
        flag = True
    if np.floor(i/size) % 2 == 1 and size - 1 - i%size  == rank:
        flag = True
    return flag

def k2_in_parallel(D,node_order,comm,rank,size,u=2):
    n = D.shape[1]
    assert len(node_order) == n, "Node order is not correct length.  It should have length %r" % n
    m = D.shape[0]
    attribute_values = vals_of_attributes(D,n)

    df = pd.DataFrame(D)
    OKToProceed = False
    parents = {}

    #selecting_job_time = 0
    #calculation_time = 0

    for i in xrange(n):
        #a = time.time()
        #mjob = my_job(i,rank,size)
        #b = time.time()
        #selecting_job_time += b-a
        if my_job(i,rank,size) == True:
            #a = time.time()
            OKToProceed = False
            pi = []
            pred = node_order[0:i]
            P_old = f(node_order[i],pi,attribute_values,df)
            if len(pred) > 0:
                OKToProceed = True
            while (OKToProceed == True and len(pi) < u):
                iters = [item for item in pred if item not in pi]
                if len(iters) > 0:
                    f_to_max = {};
                    for z_hat in iters:
                        f_to_max[z_hat] = f(node_order[i],pi+[z_hat],attribute_values,df)
                    z = max(f_to_max.iteritems(), key=operator.itemgetter(1))[0]
                    P_new = f_to_max[z]
                    if P_new > P_old:
                        P_old = P_new
                        pi = pi+[z]
                    else:
                        OKToProceed = False
                else:
                    OKToProceed = False
            parents[node_order[i]] = pi
            #b = time.time()
            #calculation_time += b-a

    # sending parents back to node 0 for sorting and printing
    #print "node ", rank, " spent ", selecting_job_time, " seconds selecting jobs"
    #print "node ", rank, " spent ", calculation_time, " seconds calculating parent sets"

    #comm.barrier()
    #a = MPI.Wtime()
    p = comm.gather(parents, root = 0)
    #comm.barrier()
    #b = MPI.Wtime()

    if rank == 0:
        #print "nodes collectively spent ", b-a, " seconds gathering the output"
    # gather returns a list - converting to a single dictionary
        parents = {}

        #a = time.time()
        for i in range(len(p)):
            parents.update(p[i])
        #b = time.time()
        #print "node 0 spent ", b-a, " seconds updating the dictionaries"

        print parents
        return parents
    


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #device = pycuda.autoinit.device.pci_bus_id()
    #node = MPI.Get_processor_name()
    
    if rank == 0:
        D = np.random.binomial(1,0.9,size=(1000,100))
        node_order = list(range(100))
    else:
        D = None
        node_order = None

    D = comm.bcast(D, root=0)
    node_order = comm.bcast(node_order, root = 0)

    comm.barrier()
    start = MPI.Wtime()
    k2_in_parallel(D,node_order,comm,rank,size,u=10)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        print "Parallel Computing Time: ", end-start

    comm.barrier()
    start = MPI.Wtime()
    oldparallel.k2_in_parallel(D,node_order,comm,rank,size,u=10)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        print "Old Parallel Computing Time: ", end-start

        serial_start = time.time()
        print serialv.k2(D,node_order, u=10)
        serial_end = time.time()
        print "Serial Computing Time: ", serial_end-serial_start