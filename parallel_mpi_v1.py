from __future__ import division
import numpy as np
import itertools
import pandas as pd
import math
import operator
import time
from mpi4py import MPI
import sys

import jodys_serial_v2 as serialv

def vals_of_attributes(D,n):
    output = []
    for i in xrange(n):
        output.append(list(np.unique(D[:,i])))
    return output

def alpha(df, mask):
    _df = df
    for combo in mask:
        _df = _df[_df[combo[0]] == combo[1]] 
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

    for i in xrange(n):
        if my_job(i,rank,size) == True:
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

    # sending parents back to node 0 for sorting and printing
    if rank == 0:
        for i in xrange(1,size):
            new_parents = comm.recv(source = i)
            parents.update(new_parents)
        # print parents
        return parents

    else:
        comm.send(parents,dest = 0)

    



if __name__ == "__main__":

    n = int(sys.argv[1])

    np.random.seed(42)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    #device = pycuda.autoinit.device.pci_bus_id()
    #node = MPI.Get_processor_name()  

    if rank == 0:
        timestoprint = []
        D = np.random.binomial(1,0.9,size=(100,n))
        node_order = list(range(n))
    else:
        D = None
        node_order = None

    D = comm.bcast(D, root=0)
    node_order = comm.bcast(node_order, root = 0)

    comm.barrier()
    start = MPI.Wtime()
    k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print timestoprint
