# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:44:21 2018

@author: Administrator
"""

with open('98input.csv') as f:#原始序列
    with open('piece_label.csv','w') as fseq:#片段所属编号
        with open('piece.csv','w') as fseq1:#片段
             with open('piece_num.csv','w') as fseq2:#每条序列切断条数
                 for (num,value) in enumerate(f):
                     s=value
                     a=50 #切片长度
#                     print(len(s))
                     fseq2.write(str(len(s)-a)+"\n")                          
                     for i in range(len(s)-a):
#                         print(s[i:i+60])  
                         fseq1.write(s[i:i+a]+"\n")
#                         print(str(num)+"\n")
                         n=str(s[i:i+a]+"\n")
                         i=i+1
                         fseq.write(str(num+1)+"\n")
#with open('98input.csv') as f:#原始序列
#    with open('seq98.csv','w') as fseq:#片段所属编号
#        with open('seq_1.csv','w') as fseq1:#片段
#             with open('seq_2.csv','w') as fseq2:#每条序列切断条数
#                 n=50 #n为片段长度
#            
#                 for (num,value) in enumerate(f):
#                     s=value
##                     print(len(s))
#                     fseq2.write(str(len(s)-n)+"\n")                          
#                     for i in range(len(s)-n):
##                         print(s[i:i+60])  
#                         fseq1.write(s[i:i+n]+"\n")
##                         print(str(num)+"\n")
#                         n=str(s[i:i+n]+"\n")
#                         i=i+1
#                         fseq.write(str(num+1)+"\n")