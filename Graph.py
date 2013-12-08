# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

Nfeatures = [3,5,10,20,30,40,50,60,70,80,90,100]
Serial  = [0.025,0.07,0.15,7.28,36.96,138.68,374.10,480.20,872,1266,1832,2567]
CUDA = [0.018,0.035,0.09,0.74,2.26,4.91,7.69,12.68,17.15,24.56,31.86,36.17]

# <codecell>

plt.plot(Nfeatures,Serial,'+-', label = 'Serial')
plt.plot(Nfeatures,CUDA,'+-',label = 'CUDA')
plt.xlabel('Number of Features')
plt.ylabel('Running Time (in seconds)')
plt.legend()
plt.title('Serial/Parallel Running Times for the K2-Algorithm')
plt.show()

# <codecell>

Nlines = [100,500,1000,5000,10000,50000,100000,200000,400000,500000]
Times_Lines = [0.49,0.56,0.61,2.61,4.02,19.97,39.36,76,154,191]

# <codecell>

plt.plot(Nlines,Times_Lines)
plt.xlabel('Number of Cases')
plt.ylabel('Running Time (in seconds)')
#.legend()
plt.title('Serial Time function of the Number of Cases')
plt.show()

# <codecell>

plt.plot(Nfeatures,Serial,'+-', label = 'Serial')
plt.xlabel('Number of Features')
plt.ylabel('Running Time (in seconds)')
#plt.legend()
plt.title('Serial Time function of the Number of Features')
plt.show()

# <codecell>


