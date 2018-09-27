# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np  
from sklearn.decomposition import PCA

for i in range(10,200,1):   #100次循环      
    test_index=i  #  测试次数
    X=np.loadtxt('F:/pca/98/98_spare.csv',delimiter=',') 
    X=np.array(X)
    #X=X.reshape(96,400)      
    pca = PCA(n_components= test_index)  
    pca.fit(X)  
    XPCA = pca.transform(X)
    #print (XPCA) 
    frame=pd.DataFrame(XPCA)
    frame.to_excel('F:/pca/98_spare_pca_'+str(test_index)+'.xlsx',header=None,index=None)