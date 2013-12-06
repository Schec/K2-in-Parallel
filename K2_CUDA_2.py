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

f_source='''
__global__ void my_f(int* d_combinations, float* d_res, int* d_ll,int* d_DF, int height) {
  
  int indi = blockIdx.x * blockDim.x + threadIdx.x;

  if(indi<height){
  float alpha_0 = 0;
  float alpha_1 = 0;
  float S3_0 = 0;
  float S3_1 = 0;
  int counter = 0;  
    #for $a in range($nrow)
      #for $b in range($ncol)
        if (d_DF[$a*$ncol+$b] == d_combinations[indi*$ncol+$b]){
	  counter += 1;
        }
      #end for
      if (counter==$ncol){
        if(d_ll[$a]==0){
          alpha_0 +=  1;
	  S3_0 += log(alpha_0);
        }
        if(d_ll[$a]==1){
          alpha_1 +=  1;
	  S3_1 += log(alpha_1);
        }
      }
      counter = 0;
    #end for

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

def nvcc_compile(string, function):
  #print string;
  module = nvcc.SourceModule(string, options=['--ptxas-options=-v'], cache_dir=False)
  return module.get_function(function)

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

def f_second(i,pi,attribute_values,df):

    #-----------------CUDA
    nfeat = df[pi].shape[1]
    phi_i_ = [attribute_values[item] for item in pi]
    combinations = np.array(list(itertools.product(*phi_i_)), dtype=np.int32)		
    combinations = np.array(combinations.ravel().tolist(), dtype=np.int32)

    ##Threads
    nf = df[pi].shape[1]
    height = np.int32(combinations.shape[0])
    block_x = np.int(height/nf)
    blocksize = (block_x,1,1)
    gridsize = (1,1)

    ##Kernel
    h_DF = np.array(df[pi], dtype = np.int32)
    h_DF = np.array(h_DF.ravel().tolist(), dtype=np.int32)
    h_ll = np.array(df[i], dtype=np.int32)
    h_res = np.zeros(height/nf,dtype=np.float32)
    d_combinations = gpu.to_gpu(combinations)
    d_res = gpu.to_gpu(h_res)
    d_ll = gpu.to_gpu(h_ll)
    d_DF = gpu.to_gpu(h_DF)
    f_kernel(d_combinations,d_res,d_ll, d_DF, height, block=blocksize,grid=gridsize)
                
    ress = d_res.get()
    ress = list(ress)
 
    return sum(ress)

if __name__ == '__main__':
  #aEH93 = np.array([[1,0,0],[1,1,1],[0,0,1],[1,1,1],[0,0,0],[0,1,1],[1,1,1],[0,0,0],[1,1,1],[0,0,0]])
  n_features = 5
  np.random.seed(10)
  D = np.random.binomial(1,0.7, size=(100,n_features))
  node_order = range(n_features)
  u = n_features-1
  start = time.time()
  #res = k2(D,range(n_features),n_features-1)
  #-----K2
  n = D.shape[1]
  m = D.shape[0]
  attribute_values = vals_of_attributes(D,n)

  df = pd.DataFrame(D)
  OKToProceed = False
  parents = []

  for i in xrange(n):
    OKToProceed = False
    pi = []
    pred = node_order[0:i]
    P_old = f(i,pi,attribute_values,df)
    if len(pred) > 0:
      OKToProceed = True
    while (OKToProceed == True and len(pi) < u):
      iters = [item for item in pred if item not in pi]
      if len(iters) > 0:
        f_to_max = {};
	#Cheetah Variables
    	template = Template(f_source)
    	template.nrow = df[pi].shape[0]
    	nfeat = df[pi].shape[1]+1
    	template.ncol = nfeat
    	f_kernel = nvcc_compile(template, "my_f")
        for z_hat in iters:
          f_to_max[z_hat] = f_second(i,pi+[z_hat],attribute_values,df)
	  #print 'i = %i, val = %f'% (i,f_to_max[z_hat])
        z = max(f_to_max.iteritems(), key=operator.itemgetter(1))[0]
        P_new = f_to_max[z]
        if P_new > P_old:
          P_old = P_new
          pi = pi+[z]
        else:
          OKToProceed = False
      else:
        OKToProceed = False
      parents.append(pi)

  print parents
  end = time.time()
  print 'Total Processing Time = %f'% (end-start)

