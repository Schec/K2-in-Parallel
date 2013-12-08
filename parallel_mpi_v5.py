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
import parallel_mpi_v4 as v4
import parallel_mpi_v3 as v3
import parallel_mpi_v2 as v2
import parallel_mpi_v1 as v1

def vals_of_attributes(D,n):
    output = []
    for i in xrange(n):
        output.append(list(np.unique(D[:, i])))
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

def find_all_jobs(i,rank,size):
    p1 = i[np.floor(i/size) % 2 == 0]
    p1 =  p1[p1%size == rank]
    p2 = i[np.floor(i/size) % 2 == 1]
    p2 =  p2[size - 1 - p2%size == rank]
    return sorted(list(p1) + list(p2), reverse = True)

def parent_set(i,node_order,attribute_values,df,u=2):
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
        return pi

def k2_in_parallel(D,node_order,comm,rank,size,u=2):
    status = MPI.Status()
    n = D.shape[1]
    assert len(node_order) == n, "Node order is not correct length.  It should have length %r" % n
    m = D.shape[0]
    attribute_values = vals_of_attributes(D,n)

    # we'll need this constant later for message sizes
    lsig = int(np.ceil(n/(2*size)))

    df = pd.DataFrame(D)
    OKToProceed = False
    parents = {}

    #selecting_job_time = 0
    #calculation_time = 0
    all_i = find_all_jobs(np.arange(n),rank,size)

    friends = range(rank + 1, size) + range(rank)
    friend_in_need = np.array([-1], dtype=np.int32)

    # this is the signal we'll send to friends who ask for work to let them know that we don't have any for them
    done = np.array([-2], dtype = np.int32)

    firstchunk = all_i[0:int(3/4*len(all_i))]
    secondchunk = all_i[int(3/4*len(all_i)):len(all_i)]

    for i in firstchunk:
        parents[node_order[i]] = parent_set(i, node_order, attribute_values, df, u)

    lsec = len(secondchunk)

    while lsec > 0:

        req = comm.Irecv(friend_in_need, source = MPI.ANY_SOURCE) # this needs to be non-blocking, asynchronous communication -- no pickle option available
        if req.Test(status = status) == False:
            req.Cancel()

        # deal with the friend in need
        if not friend_in_need == -1:

            # identify the friend who sent the message
            if friend_in_need == -2:
                friend =  status.Get_source()
            else:
                friend = friend_in_need

            # this friend asked for work, so don't ask for work from this friend later
            if friend in friends:
                friends.remove(friend)

            # send done message if we don't have a lot of work left
            if lsec < 4:
                comm.Send(done, dest = friend)

            # don't send any work if the friend just sent that he's done - he'll ask again if he actually wants work
            # send half of the remaining work if there is enough left,
            if lsec >= 4 and not friend_in_need == -2:
                # build the message
                a = list(secondchunk[int(np.ceil(1/2*lsec)):lsec])
                # pad the message with zeros (for consistent-sized messages)
                b = list(np.zeros(lsig-len(a)))
                # send the message
                comm.Send(np.array(a+b, dtype=np.int32), dest=friend)
                # update my own chunk of work
                secondchunk = secondchunk[0:int(np.ceil(1/2*lsec))]

            friend_in_need =  np.array([-1], dtype=np.int32)


        # whether you sent to a friend or not choose the next element to calculate
        i = secondchunk.pop(0) 
        parents[node_order[i]] = parent_set(i, node_order, attribute_values, df, u)

        # update lsec to see if we should go again
        lsec = len(secondchunk)

    # send done signals toeverybody else
    for f in friends:
        comm.Send(done, dest = f)

    # nodes that are done with work ask their neighbors for work units
    # you could receive up to np.floor(n/(2*size)) work units
    signal = np.zeros(shape = lsig, dtype = np.int32)

    while(len(friends) > 0):
        destination = friends.pop(0)
        mess = np.array([rank], dtype=np.int32)
        signal = np.zeros(shape = lsig, dtype = np.int32)
        # check to see whether this node sent you a done message
        req = comm.Irecv(signal, source = destination) 
        if req.Test(status = status) == False:
            req.Cancel()
            # if not, send him your rank and hope for work back!
            comm.Send(mess, dest = destination) #sending rank as message eliminates need for Get_source()
            comm.Recv(signal, source = destination)

        # get any work from signal that exists
        secondchunk = signal[signal > 0]
        for i in secondchunk:
            parents[node_order[i]] = parent_set(i, node_order, attribute_values, df, u)

    # sending parents back to node 0 for sorting and printing
    p = comm.gather(parents, root = 0)
    
    if rank == 0:
    # gather returns a list - converting to a single dictionary
        parents = {}

        for i in range(len(p)):
            parents.update(p[i])

        #print parents
        return parents

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
    v1.k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print "v1 complete"

    comm.barrier()
    start = MPI.Wtime()
    v2.k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print "v2 complete"

    comm.barrier()
    start = MPI.Wtime()
    v3.k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print "v3 complete"

    comm.barrier()
    start = MPI.Wtime()
    v4.k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print "v4 complete"

    comm.barrier()
    start = MPI.Wtime()
    k2_in_parallel(D,node_order,comm,rank,size,u=n-1)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        timestoprint.append(end-start)
        print "v5 complete"
        print timestoprint
