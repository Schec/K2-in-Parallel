# -*- coding: utf-8 -*-

import numpy as np
import math
import collections
import itertools
import pandas as pd
from __future__ import division

#Parameters
x1 = [1,1,0,1,0,0,1,0,1,0]
x2 = [0,1,0,1,0,1,1,0,1]
x3 = [0,1,1,1,0,1,1,0,1,0]
r = [2,2,2]
n = 3
u = 2
D = np.array([[1,0,0],[1,1,1],[0,0,1],[1,1,1],[0,0,0],[0,1,1],[1,1,1],[0,0,0],[1,1,1],[0,0,0]])
D = pd.DataFrame(D)
pi = {}
Pred = [[],[0],[0,1]]

for i in xrange(3):
    pi[str(i)] = []

for i in xrange(3):
    #print '-------------------'+str(i)
    
    #Initialization of P_old
    unique_vals = list(set(D[i]))
    alphas = []
    for ele in unique_vals:
        alphas.append(sum([x==ele for x in D[i]]))
    N = sum(alphas)
    P_old = math.factorial(r[i]-1)/math.factorial(N+r[i]-1)*np.product([math.factorial(x) for x in alphas])
    #print 'P_old = %f' % P_old
    
    
    #START
    OKToProceed = True
    the_list = []
    while OKToProceed and len(pi[str(i)])<u and len(Pred[i])>0:
        vals = []
        Dyn_Pred = [x for x in Pred[i] if x not in pi[str(i)]]
        for pred in Dyn_Pred:
            unique = list(set(D[pred]))
            temp = [x for x in the_list]
            temp.append(unique)
            combinations = list(itertools.product(*temp))
            val1 = 1
            temp_nodes = [x for x in pi[str(i)]]
            temp_nodes.append(pred)
            for ele in combinations:
                temp_D = D
                #print '----'+str(pred)
                #print ele
                for l in range(len(ele)):
                    temp_D = temp_D[temp_D[temp_nodes[l]]==ele[l]]
                #print i
                #print temp_D
                SS = []
                for uval in unique_vals:
                    SS.append(sum(temp_D[i]==uval))
                NN = sum(SS)
                prod_fact = np.prod([math.factorial(x) for x in SS])
                val1 *= math.factorial(r[i]-1)/math.factorial(NN+r[i]-1)*prod_fact
            vals.append(val1)
        #print vals
        if len(vals)>0:
            P_new = np.max(vals)
            p = [ww for ww, j in enumerate(vals) if j == P_new]
            pos = p[0]
            if P_new > P_old:
                P_old = P_new
                pi[str(i)].append(Pred[i][pos])
                the_list.append(list(set(D[pred])))
            else:
                OKToProceed = False
        else:
            OKToProceed = False
    if len(pi[str(i)])>0:
        for ele in pi[str(i)]:
            print 'Node: x'+str(i+1)+', Parents of x'+str(i+1)+': x'+str(ele+1)
    else:
        print 'Node: x'+str(i+1)+' has no parents!!'
        
