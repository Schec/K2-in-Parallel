import numpy as np
import itertools
import pandas as pd
import math
import operator
import time
from mpi4py import MPI
import matplotlib.pyplot as plt

import parallel_mpi_v2 as v1
import parallel_mpi_v3 as v2

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#device = pycuda.autoinit.device.pci_bus_id()
#node = MPI.Get_processor_name()

v1_times = []
v2_times = []
ns = list(range(10,101,10))

for n in ns:

    if rank == 0:
        D = np.random.binomial(1,0.9,size=(10000,n))
        node_order = list(range(n))
    else:
        D = None
        node_order = None

    D = comm.bcast(D, root=0)
    node_order = comm.bcast(node_order, root = 0)

    comm.barrier()
    start = MPI.Wtime()
    v1.k2_in_parallel(D,node_order,comm,rank,size,u=5)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        v1_times.append(end-start)

    comm.barrier()
    start = MPI.Wtime()
    v2.k2_in_parallel(D,node_order,comm,rank,size,u=5)
    comm.barrier()
    end = MPI.Wtime()
    if rank == 0:
        v2_times.append(end-start)

if rank ==0:
    plt.plot(ns,v1_times,'r',label='V2')
    plt.plot(ns,v2_times,'b',label='V3')
    plt.legend()
    plt.title('Comparison of Send/Recv and Gather Versions of MPI')
    plt.xlabel('Number of Features')
    plt.ylabel('Time (seconds)')
    plt.show()