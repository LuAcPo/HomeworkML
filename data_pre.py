import numpy as np
import datetime
import time
import torch

def get_d0():
    adj=np.load('./cora_numpy/adjacency_matrix.npy')
    adj=np.logical_or(adj,adj.T)
    np.save('./data/d0.npy',np.array(adj,dtype=np.int16))
    return adj
def get_d1():
    d0=np.load('./data/d0.npy')
    print(np.max(d0))
    print(np.min(d0))
    temp_d1=d0@d0
    mark_d0=d0*temp_d1
    d1=temp_d1-mark_d0-np.diag(np.sum(d0,axis=1))#最小距离为2的点，路径个数
    
    print(np.max(d1))
    print(np.min(d1))
    np.save('./data/mk.npy',mark_d0)#
    np.save('./data/d1.npy',d1)
    return d1
def get_mask():
    d0=np.load('./data/d0.npy')
    d1=np.load('./data/d1.npy')
    mask=((d0+d1)>0)
    # print(np.count_nonzero(mask))
    np.save('./data/mask.npy',mask)
    # print(mask)
    return mask


def finail_martix(temperature):
    d0=np.load('./data/d0.npy')
    d1=np.load('./data/d1.npy')
    mk=np.load('./data/mk.npy')
    deg=np.sum(d0,axis=1)
    setU=(deg[np.newaxis,:]+deg[:,np.newaxis]-d1+1)# A--C--B A-/-B set(A U B)
    adjd1=np.exp(-(1-d1/setU)/temperature)*(d1>0)
    adjd0=np.exp((mk/(deg[:,np.newaxis]+1)-1)/temperature)*(mk>0)
    print(np.max(adjd1))
    print(np.max(d1/(setU)))
    print(np.sum(adjd1))
    print(np.max(adjd0))
    print(np.sum(adjd0))
    result=adjd0+adjd1+d0
    print(np.max(result))
    np.save('./data/dataT'+str(temperature)+'.npy',result)
    np.save('data/mask.npy',result>0)
if __name__ =='__main__':
    # get_d0()
    # get_d1()
    # get_mask()
    finail_martix(1)
    finail_martix(2)