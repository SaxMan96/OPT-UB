import numpy as np
import scipy
import matplotlib.pyplot as plt
from utilities import *


def gen_dataset(n, separable=True):
    m1 = [0. ,0.]
    s1 = [[1, -0.9],[-0.9,1]]
    c1 = np.random.multivariate_normal(m1,s1,n//2)
#    x1 = c1[:,0]
#    y1 = c1[:,1]

    if separable == True:
        m2 = [3. ,6.]
        s2 = s1
    if separable == False:
        m2 = [1. ,2.]
        s2 = [[1, 0],[0,1]]
    c2 = np.random.multivariate_normal(m2,s2,n//2)
#    x2 = c2[:,0]
#    y2 = c2[:,1]
    x = np.concatenate((c1,c2), axis=0).T
    y = np.concatenate(([1]*(n//2), [-1]*(n//2)), axis=0)
    return x, y

def prob_dual_gen(x, y, K):
    def prob_dual(n):
        return prob_dual_complete(n, x, y, K)
    return prob_dual

def prob_dual_complete(n, x, y, K):

    Y = np.diag(y)

    p = 1
    m = 2*n
    N = n + p + 2*m
    #K = 100.
    A = y.reshape(n,1)
    b = np.array([0])
    G = -Y @ x.T @x @ Y
    g = np.array([1.]*n)
    C = np.concatenate((np.eye(n),-np.eye(n)), axis=0).T
    d = np.concatenate(([0.]*n, [-K]*n))
    #z = np.zeros(N)
    #z = np.array([1.]*N)
    z = np.random.rand(N)
    z[-2*m:] = 1.
    print('x:%s Y:%s C:%s d:%s' %(x.shape, Y.shape,  C.shape, d.shape))
    return z, G, A, C, g, b, d, n, m, p

n=100
separable = True
x, y = gen_dataset(n, separable)
print( x.shape, y.shape)
#K_list = [1,10,100,1000,int(1e12)]
K_list = range(1,10)
for K  in K_list:
    print('K:%s separable:%s|'%(K, separable), end = ' ')
    prob_dual = prob_dual_gen(x, y, K)
    print(type(prob_dual))
    optimization(n, prob_dual, s0, np.linalg.solve, cond_nb=False)
