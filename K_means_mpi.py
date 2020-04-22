#! /usr/bin/env python
# -*- coding: utf-8 -*-

 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from numpy.random import randint
import time
from math import sqrt
from collections import defaultdict
from sklearn.datasets import load_iris
from IPython.core.debugger import set_trace
Comm = MPI.COMM_WORLD
rank = Comm.Get_rank()
size= Comm.Get_size()
data_type = int # our data type is integer for this example
root = 0
K = 5
#loading the Iris data
Hasmap  =  {'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}
def scatter_plot(Data,cent, K_component,dim=2, ):
    plt.figure(figsize=(15,7))
    k_colore = ['lightgreen','orange','lightblue','black', 'darkblue','brown']
    k_colore = k_colore[:len(cent)]
    marker = ['s','o','v','x','p','*']
    marker = marker[:len(cent)]
    for num, k in enumerate(list(K_component)):
        plt.scatter(
            Data[K_component[k]][:,0],Data[K_component[k]][:,1],
            s=50, c=k_colore[num],
            marker=marker[k], edgecolor='black',
            label=f"cluster {num}"
        )

        # plot the centroids
        plt.scatter(
            cent[k][0], cent[k][1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids'
        )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()

def sq_eucl_dist(x,y):
    """
    Perform the square euclidien Distance
    """
    sq = pow((x-y),2)
    return np.sum(sq)

def Weizfeld_median(A,max_iter=100):
    "Return the geometric median "
    mu_0 = mean_(A)
    for it in range(max_iter):
        R = np.array([x*np.sqrt(sq_eucl_dist(x,mu_0))**(-1) for x in A])
        t = np.array(list(sum(x) for x in zip(*R)))
        mu =t/ sum(np.sqrt(sq_eucl_dist(x,mu_0))**(-1) for x in A)
        if abs(sum(mu_0-mu)) < 10**(-10):
            return mu
        mu_0 = mu
    return mu

def mean_(x):
    return np.mean(x,axis=0)


def cent_init(A, K):

    cent_list = defaultdict()
    index = np.random.choice(A.shape[0],K, replace=False)
    for num, k in enumerate(index):
        cent_list[num] = A[k]
    return cent_list


def comput_centroid( A, cluster_element,out_cent, geom_median=False, max_iter=1000):
    centroids = out_cent.copy()
    if geom_median:
        for centroid, element in cluster_element.items():
            if len(element)>1:

                mu = Weizfeld_median(A[element], max_iter)
                centroids[centroid]= mu
        return centroids

    for centroid, element in cluster_element.items():
        if len(element)>1:
            mu = mean_(A[element])
            centroids[centroid]= mu
    return centroids




def k_mean(A, init):
    kluster_list =  list(init.keys())
    cluster_element = defaultdict(list)
    J_clust = 0
    for num, el in enumerate(A):
        list_dist = []
        for kluster in kluster_list:
            list_dist.append(sq_eucl_dist(el,init[kluster]))
        close_clt = np.argmin(list_dist)
        J_clust += min(list_dist)
        cluster_element[kluster_list[close_clt]].append(num)
    for k in kluster_list:
        if k not in cluster_element:
            cluster_element[k].append(0)
    return cluster_element, J_clust/(A.shape[0])

def aggreg(cent,K,total_rank):
    global_cent = defaultdict()
    for i in range(K):
        global_cent[i] = np.zeros(1,dtype=float)
    for cen in cent:
        for key, value in cen.items():
            global_cent[key] = global_cent[key] + value/total_rank
    return global_cent

def test_equal(cent1,cent2):
     sum_to = 0
     for key in cent1:
         su= np.sum(abs(cent1[key]-cent2[key]))
         sum_to +=su
     return sum_to

if rank == root:
    data = load_iris()
    # X,y = data.data , data.target
    X, y = make_blobs(n_samples=3000, centers=5, cluster_std=0.60, random_state=0)
    np.random.shuffle(X)
    init_cent = cent_init(X,K)
    print(f"root{rank}--init_centroid{init_cent}")
    # print(init_cent)
    data = np.array_split(X, size)
else:
    data = None
    init_cent = None
cents = Comm.bcast(init_cent,root=0)
data = Comm.scatter(data, root=0)
converge = Comm.bcast(True, root=0)

total_distortion = []
t =0
while converge:
    cent_to = cents.copy()
    final_groupe , J_clust = k_mean(data,cents)
    cent_new    = comput_centroid(data,final_groupe,cents)
    local_cent  = Comm.allgather(cent_new)
    global_cent =  aggreg(local_cent,K,size)
    sum_cent    = None
    if t > 0:
        s = 0
        for k in range(K):
            s = s + len(set(final_groupe[k])-(set(init_groupe[k])))
        sum_cent = Comm.allreduce(s)
    if rank==root:
        print('final_groupe[1',sum_cent)

    Comm.barrier()
    if sum_cent != None and sum_cent == 0:
        converge = False
        print('terminate')
    cents = global_cent.copy()
    init_groupe = final_groupe.copy()
    t += 1

if rank==root:
    plot_cent,J_clust = k_mean(X,global_cent)
    scatter_plot(X,global_cent,plot_cent)
 
