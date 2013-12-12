from __future__ import division
import numpy as np
import itertools
import pandas as pd
import operator
from mpi4py import MPI
import sys
import argparse


def vals_of_attributes(D, n):
    output = []
    for i in xrange(n):
        output.append(list(np.unique(D[:, i])))
    return output


def alpha(df, mask):
    _df = df
    for combo in mask:
        _df = _df[_df[combo[0]] == combo[1]]
    return len(_df)


def f(i, pi, attribute_values, df):

    phi_i_ = [attribute_values[item] for item in pi]
    if len(phi_i_) == 1:
        phi_i = [[item] for item in phi_i_[0]]
    else:
        phi_i = list(itertools.product(*phi_i_))

    # bug fix: phi_i might contain empty tuple (), which shouldn't count in q_I
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

    for j in js:

        # initializing mask to send to alpha
        if j == 'special':
            mask = []
        else:
            mask = zip(pi, phi_i[j])

        # initializing counts that will increase with alphas
        N_ij = 0
        #inner_product = 1
        inner_product = 0

        for k in xrange(r_i):
            # adjusting mask for each k
            mask_with_k = mask + [[i, V_i[k]]]
            alpha_ijk = alpha(df, mask_with_k)
            N_ij += alpha_ijk
            #inner_product = inner_product*math.factorial(alpha_ijk)
            inner_product = inner_product + np.sum([np.log(b) for b in range(
                1, alpha_ijk + 1)])
        #denominator = math.factorial(N_ij + r_i - 1)
        denominator = np.sum([np.log(b) for b in range(1, N_ij + r_i)])
        #product = product*(numerator/denominator)*inner_product
        product = product + numerator - denominator + inner_product
    return product


def find_all_jobs(i, rank, size):
    p1 = i[np.floor(i / size) % 2 == 0]
    p1 = p1[p1 % size == rank]
    p2 = i[np.floor(i / size) % 2 == 1]
    p2 = p2[size - 1 - p2 % size == rank]
    return sorted(list(p1) + list(p2))


def parent_set(i, node_order, attribute_values, df, u=2):
        OKToProceed = False
        pi = []
        pred = node_order[0:i]
        P_old = f(node_order[i], pi, attribute_values, df)
        if len(pred) > 0:
            OKToProceed = True
        while (OKToProceed is True and len(pi) < u):
            iters = [item for item in pred if item not in pi]
            if len(iters) > 0:
                f_to_max = {}
                for z_hat in iters:
                    f_to_max[z_hat] = f(node_order[i], pi + [z_hat],
                        attribute_values, df)
                z = max(f_to_max.iteritems(), key=operator.itemgetter(1))[0]
                P_new = f_to_max[z]
                if P_new > P_old:
                    P_old = P_new
                    pi = pi + [z]
                else:
                    OKToProceed = False
            else:
                OKToProceed = False
        return pi


def k2_in_parallel(D, node_order, comm, rank, size, u=2):
    status = MPI.Status()
    n = D.shape[1]
    assert len(node_order) == n, ("Node order is not correct length."
        "  It should have length %r" % n)
    attribute_values = vals_of_attributes(D, n)

    # we'll need this constant later for message sizes
    lsig = int(np.ceil(n / (2 * size)))

    df = pd.DataFrame(D)
    parents = {}

    #selecting_job_time = 0
    #calculation_time = 0

    all_i = find_all_jobs(np.arange(n, dtype=np.int32), rank, size)

    friends = range(rank + 1, size) + range(rank)
    friend_in_need = np.array([-1], dtype=np.int32)

    # this is the signal we'll send to let friends know that we don't have work
    done = np.array([-2], dtype=np.int32)

    lall = len(all_i)

    friends_who_are_done = []
    friends_who_know_im_done = []

    while lall > 0:

        # this needs to be non-blocking -- no pickle option available
        req = comm.Irecv(friend_in_need, source=MPI.ANY_SOURCE)
        if req.Test(status=status) is False:
            req.Cancel()

        # deal with the friend in need
        if not friend_in_need == -1:

            # identify the friend who sent the message
            if friend_in_need == -2:
                friend = status.Get_source()
            else:
                friend = friend_in_need[0]

            friends_who_are_done.append(friend)

            # send done message if we don't have a lot of work left
            if lall < 4 and friend not in friends_who_know_im_done:
                comm.Isend(done, dest=friend)
                friends_who_know_im_done.append(friend)

            # don't send any work if the friend just sent that he's done
            # send half of the remaining work if there is enough left,
            if lall >= 4 and not friend_in_need == -2:

                # build the message
                mess = all_i[int(np.ceil(1 / 2 * lall)):lall]

                # pad the message with zeros (for consistent-sized messages)
                pad = np.zeros(lsig - len(mess), dtype=np.int32)
                mess = np.concatenate((mess, pad), axis=1)

                # send the message
                comm.Send(mess, dest=friend)

                # update my own chunk of work
                all_i = all_i[0:int(np.ceil(1 / 2 * lall))]

                # update length of chunk of work
                lall = len(all_i)

            friend_in_need = np.array([-1], dtype=np.int32)

        # (bottleneck fix) calculate only if friend_in_need == -1
        else:
            # choose the next element to calculate
            i = all_i.pop(0)
            parents[node_order[i]] = parent_set(
                i, node_order, attribute_values, df, u)

            # update length of work
            lall = lall - 1

    # send done signals to everybody else
    for f in friends:
        if f not in friends_who_know_im_done:
            comm.Send(done, dest=f)

    friends = [bud for bud in friends if bud not in friends_who_are_done]

    # nodes that are done with work ask their neighbors for work units
    # you could receive up to np.floor(n/(2*size)) work units

    status = MPI.Status()

    while(len(friends) > 0):
        destination = friends.pop(0)
        mess = np.array([rank], dtype=np.int32)
        signal = np.zeros(shape=lsig, dtype=np.int32)

        # check to see whether this node sent you a done message
        req = comm.Irecv(signal, source=destination)
        if req.Test(status=status) is False:
            req.Cancel()
             # if not, send him your rank and hope for work back!
            comm.Send(mess, dest=destination)
            comm.Recv(signal, source=destination)

        # get any work from signal that exists
        all_i = signal[signal > 0]
        for i in all_i:
            parents[node_order[i]] = parent_set(
                i, node_order, attribute_values, df, u)

    # sending parents back to node 0 for sorting and printing
    p = comm.gather(parents, root=0)

    if rank == 0:
    # gather returns a list - converting to a single dictionary
        parents = {}

        for i in range(len(p)):
            parents.update(p[i])

        return parents

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''K2 In Serial:  Calculates
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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    u = args.u

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

    elif not args.D == None:
        #if rank == 0:
            # "Reading in array D"
        D = np.loadtxt(open(args.D))
        if args.node_order != None:
            node_order = args.node_order
        else:
            #if rank == 0:
                #print "Determining node order"
            n = np.int32(D.shape[1])
            node_order = list(range(n))

    else:
        if rank == 0:
            print "Incorrect usage. Use --help to display help."
        sys.exit()

    #if rank == 0:
        #print "Calculating Parent sets"
    comm.barrier()
    start = MPI.Wtime()
    parents = k2_in_parallel(D,node_order,comm,rank,size,u=u)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        #print "Parallel computing time", end-start
        #print parents
        print "V4", n, m, size, end-start
