from __future__ import division
import numpy as np
import itertools
import pandas as pd
import math
import operator
import time

from Cheetah.Template import Template
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.compiler as nvcc
#d_combinations[ind_i*$W+$b]
f_source='''
__global__ void my_f(int** d_combinations, float* d_res) {
  
  int ind_i = blockIdx.x * blockDim.x + threadIdx.x;

  float alpha_0 = 0;
  float alpha_1 = 0;
  float S3_0 = 0;
  float S3_1 = 0;
  int counter = 0;  
  if(ind_i<$H){
    #for $a in range($nrow)
      #for $b in range($ncol)
        if ($DF[$a][$b] == d_combinations[ind_i][$b]){
	  counter += 1;
        }
      #end for
      if (counter==$ncol){
        if($ll[$a]==0){
          alpha_0 +=  1;
	  S3_0 += log2(alpha_0);
        }
        if($ll[$a]==1){
          alpha_1 +=  1;
	  S3_1 += log2(alpha_1);
        }
      }
      counter = 0;
    #end for

  int N = alpha_0 + alpha_1;
  int r = 2;
  float S1 = 0;
  if (N>0){
    for (float n = r; n <= (N+r-1);n++){
      S1 += log2(n);
    }
  }
  d_res[ind_i] = S3_1 +S3_0-S1;
  }
}'''

def nvcc_compile(string, function):
  print string;
  module = nvcc.SourceModule(string, options=['--ptxas-options=-v'], cache_dir=False)
  return module.get_function(function)

if __name__ == '__main__':
	D = np.array([[1,0,0],[1,1,1],[0,0,1],[1,1,1],[0,0,0],[0,1,1],[1,1,1],[0,0,0],[1,1,1],[0,0,0]],dtype=np.uint8)
	n_features = 3
	node_order = range(n_features)
  
  	#-----------------CUDA
	#Cheetah Variables
	template = Template(f_source)
	template.DF = np.array(D[:,:2], dtype = np.uint8)
	template.nrow = D[:,:2].shape[0]
	template.ncol = D[:,:2].shape[1]
	template.ll = np.array(D[:,2], dtype=np.uint8)

	combinations = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.uint8)		
	template.H = combinations.shape[0]
	template.W = combinations.shape[1]
	f_kernel = nvcc_compile(template, "my_f")

	##Threads
	height = combinations.shape[0]
	blocksize = (height,1,1)
	gridsize = (1,1)

	##Kernel
	h_res = np.zeros(height,dtype=np.float32)
	
	d_combinations = gpu.to_gpu(combinations)
	print d_combinations.get()	
	d_res = gpu.to_gpu(h_res)
	f_kernel(d_combinations, d_res,block=blocksize,grid=gridsize)		      
	h_res = d_res.get()
	
        for ele in h_res:
	  print ele

