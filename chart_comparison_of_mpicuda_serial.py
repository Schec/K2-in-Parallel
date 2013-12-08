from matplotlib import pyplot as plt

n =  [5, 10, 20, 30, 40, 50, 60, 100]

serial = [0.07, 0.15, 7.28, 36.96, 138.68, 374.10, 480.20, 2567]
mpiandcuda = [0.01772904396057129, 0.07807302474975586, 0.29834699630737305, 0.8033761978149414, 1.7457571029663086, 3.0404789447784424, 4.631350994110107, 17.509130001068115]

plt.plot(n, mpiandcuda, label = 'Parallel')
plt.plot(n, serial, label = "Serial")
plt.yscale('log')
plt.title('Comparison of MPI and Cuda Script to Serial - MPI V3, 4 Nodes')
plt.legend(loc  = 2)
plt.xlabel('Number of Features')
plt.ylabel('Time')
plt.show()



