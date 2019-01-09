# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 07:03:28 2018

@author: Administrator
"""


from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import time

pipe_svc = Pipeline([('scl', StandardScaler()),
            ('clf', SVC(random_state=0))])

#param_range =[]

param_grid = [{'clf__C':[0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000] ,
                  'clf__gamma': [10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],
                  'clf__kernel': ['rbf']}]

#param_grid = [{'clf__C':[0.01,0.1,1,10,100,1000],
#                  'clf__gamma': [10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],
#                  'clf__kernel': ['rbf']}]
##网格搜索
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=100,
                  n_jobs=1)




#训练模型
time_start=time.time()
for i in range(40,100,20):   #设置字典大小为别为20,40，60...直到300
    components=i        
    X_train =np.loadtxt('zd98_feature'+str(components)+'.csv',delimiter=',')
    
    
      
    
    y_train =np.loadtxt('train_label.csv',delimiter=',')
    X_test =np.loadtxt('zd98_feature'+str(components)+'.csv',delimiter=',')
    
    
    
    y_test =np.loadtxt('test_label.csv',delimiter=',')
    
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
   
    #0.978021978022
    #{'clf__kernel': 'linear', 'clf__C': 0.1}
            
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    svm_model=clf.fit(X_train, y_train)
    
    print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    
    time_end=time.time() 
    print('totally cost',time_end-time_start)

    with open("acc.txt","a+") as f:
        f.write(str(gs.best_score_)+'\n'+str(gs.best_params_)+'\n'+str(clf.score(X_test, y_test))+'\n'+str(time_end-time_start)+'\n'+'\n')
       
#
#Test accuracy: 0.965

#保存模型
    joblib.dump(svm_model,'svm_model_zd98.model')

#加载模型
svm_model=joblib.load('svm_model_zd98.model')
 
#应用模型进行预测
result=svm_model.predict(X_test)