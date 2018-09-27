# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 12:18:15 2018

@author: cxj
"""
# coding:utf-8
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
import scipy as sp
import pandas as pd
from sklearn.linear_model import orthogonal_mp_gram
#from sklearn import preprocessing

def max_pooling(matrix):
    max_list=[]
    for i in range(matrix.shape[1]):
        max_list.append(max(matrix[:,i]))
    max_list=np.array(max_list)
    return max_list
    
def mean_pooling(matrix):
    mean_list=[]
    for i in range(matrix.shape[1]):
        mean_list.append(sum(matrix[:,i])/len(matrix[:,i]))
    mean_list=np.array(mean_list)
    return mean_list


def random_pick(some_list,probabilities):
     import random    
     x=random.uniform(0,1)
     cumulative_probability=0.0
     for item,item_probability in zip(some_list,probabilities):
         cumulative_probability+=item_probability
         if x < cumulative_probability: break
     return item


def random_pooling(matrix): 
    from sklearn.preprocessing import MinMaxScaler  
    pick_list=[]
    for i in range(matrix.shape[1]):       
        matrix_list=matrix[:,i]
        matrix_list=matrix_list.reshape(len(matrix_list),1)
        mms = MinMaxScaler(copy=True, feature_range=(0, 1))
        mms.fit(matrix_list)  
        matrix_prob=mms.transform(matrix_list)
#        matrix_temp=matrix_list -min(matrix_list)
#        matrix_seq=float(max(matrix_list) -min(matrix_list))
#        matrix_prob = matrix_temp/ matrix_seq
        matrix_probablity=matrix_prob/(matrix_prob.sum())   #每一列的概率         
#        matrix_probablity=list(matrix_probablity)
#        matrix_prob=list(matrix_prob)        
        pick_matrix=random_pick(matrix_prob,matrix_probablity)
        matrix_index=(matrix_prob.tolist()).index(pick_matrix.tolist())
        pick_list.append(matrix_list[matrix_index])    
    pick_list=np.array(pick_list)
    pick_list=pick_list.reshape(1,matrix.shape[1])
    return pick_list
    
    
def max_mean_pooling(matrix):
    max_list=[]
    mean_list=[]
    for i in range(matrix.shape[1]):
        max_list.append(max(matrix[:,i]))
        mean_list.append(sum(matrix[:,i])/len(matrix[:,i]))
    max_list=np.array(max_list)
    mean_list=np.array(mean_list)
    max_mean=(max_list+mean_list)/2
    return max_mean
   
     
def less_max(matrix_list):                
    order=sorted(list(matrix_list),reverse = True)    
    if min(order)==max(order):
            return max(order)
    else:
        less_max_list=[]
        for j in range(len(order)):
            if order[j]<max(order):
                less_max_list.append(order[j])                                                            
        less_max=max(less_max_list)
        return less_max

                  
def double_max_pooling(matrix):
    max_list=[]
    less_max_list=[]
    for i in range(matrix.shape[1]):             
        max_list.append(max(matrix[:,i]))                   
        less_max_list.append(less_max(matrix[:,i]))                
    max_list=np.array(max_list)
    less_max_list=np.array(less_max_list)
    double_max=(max_list+less_max_list)/2    
    return double_max   

class ApproximateKSVD(object):
    def __init__(self, n_components, max_iter=30, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements    字典原子个数

        max_iter:
            Maximum number of iterations   最大迭代次数

        tol:
            tolerance for error   结果容差

        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target   稀疏度，即非0系数的数量（默认值原始样本特征维度*0.1）
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs


    def _update_dict(self, X, D, gamma):       #字典更新 X为样本 D为字典  gamma为稀疏表达
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.2* X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        """
        Parameters
        ----------
        X: shape = [n_samples, n_features]
        """
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


X =np.loadtxt('317aac_piece.csv',delimiter=',')
#X= preprocessing.scale(X)
with open('317piece_num.csv') as f:#每条蛋白质序列切片片段数  96维
    c=[]
    for p in f:
        p= p.strip('\n')
#        q=float(p)
        q=int(float(p))
        c.append(q)
temp=[]    
for u in range(len(c)):
    u=int(u)
#    a=a[0:u+1]
#    print(sum(c[0:u+1]))        
    temp.append(sum(c[0:u+1]))    
p=[]   
for i in range(316):
   p.append(temp[i]-c[i]+1) 
  
for i in range(25,301):   #300次循环
#    starttime = datetime.datetime.now()
#    test_index=i+1  #  测试次数
    components=i 
    aksvd = ApproximateKSVD(n_components=components)
    dictionary = aksvd.fit(X).components_
    gamma = aksvd.transform(X)
    frame_gamma=pd.DataFrame(gamma)
    frame_gamma.to_excel('F:317svm/spare/317_gamma_spare_'+str(components)+'.xlsx',header=None,index=None)  
    d = [i for i in range(316)]  
#    max_feature=[]
    mean_feature=[]
#    random_feature=[]
#    max_mean_feature=[]
#    double_max_feature=[]     
    for i,j in enumerate(c):  #c有96条
        u=p[i]
        v=c[i]
        d[i]=(gamma[u-1:u+v-1:])    
        n = d[i]
#       print(n.shape)        
#       n=np.transpose(n)                                                
#        max_feature.append(max_pooling(n))
        mean_feature.append(mean_pooling(n))
#        random_feature.append(random_pooling(n))
#        max_mean_feature.append( max_mean_pooling(n))
#        double_max_feature.append(double_max_pooling(n))
#    max_feature= np.array(max_feature)
    mean_feature= np.array(mean_feature)
#    random_feature= np.array(random_feature)
#    max_mean_feature= np.array(max_mean_feature)
#    double_max_feature= np.array(double_max_feature)      
#    max_feature=max_feature.reshape(96,components)       
    mean_feature=mean_feature.reshape(316,components)
#    random_feature=random_feature.reshape(96,components)
#    max_mean_feature=max_mean_feature.reshape(96,components)
#    double_max_feature=double_max_feature.reshape(96,components)
#    spare_feature= np.array(spare_feature)
#    spare_feature = preprocessing.scale(spare_feature)        
#    frame=pd.DataFrame(max_feature)    
#    frame.to_excel('F:/cxj/98svm/spare_18/98_max_spare_'+str(components)+'.xlsx',header=None,index=None) 
    frame=pd.DataFrame(mean_feature)    
    frame.to_excel('F:/317svm/spare/317_mean_spare_'+str(components)+'.xlsx',header=None,index=None)
#    
#    frame=pd.DataFrame(random_feature)    
#    frame.to_excel('F:/cxj/98svm/spare_18/98_random_spare_'+str(components)+'.xlsx',header=None,index=None)
#    
#    frame=pd.DataFrame(max_mean_feature)    
#    frame.to_excel('F:/cxj/98svm/spare_18/98_max_mean_spare_'+str(components)+'.xlsx',header=None,index=None)
#    
#    frame=pd.DataFrame(double_max_feature)    
#    frame.to_excel('F:/cxj/98svm/spare_18/98_double_max_spare_'+str(components)+'.xlsx',header=None,index=None)
#    endtime = datetime.datetime.now()  
#    print (endtime - starttime)
