# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


test = open("piece.csv", "r")
line=len(test.readlines())
with open('piece.csv') as fseq1:          
         D=np.zeros((line,20),dtype=np.float)#line为切片后片段段数
         for i,n in  enumerate(fseq1):
             #s=line
             s=n
             #print (s.count("A"))
             #a=s.count("A")
             a=[] 
             b=(((s.count("A")))/50.0)
             D[i][0]=b
             b1=((float(s.count("C")))/50.0)
             D[i][1]=b1
             b2=((float(s.count("D")))/50.0)
             D[i][2]=b2
             b3=((float(s.count("E")))/50.0)
             D[i][3]=b3
             b4=((float(s.count("F")))/50.0)
             D[i][4]=b4
             b5=((float(s.count("G")))/50.0)
             D[i][5]=b5
             b6=((float(s.count("H")))/50.0)
             D[i][6]=b6
             b7=((float(s.count("I")))/50.0)
             D[i][7]=b7
             b8=((float(s.count("K")))/50.0)
             D[i][8]=b8
             b9=((float(s.count("L")))/50.0)
             D[i][9]=b9
             b10=((float(s.count("M")))/50.0)
             D[i][10]=b10
             b11=((float(s.count("N")))/50.0)
             D[i][11]=b11
             b12=((float(s.count("P")))/50.0)
             D[i][12]=b12
             b13=((float(s.count("Q")))/50.0)
             D[i][13]=b13
             b14=((float(s.count("R")))/50.0)
             D[i][14]=b14
             b15=((float(s.count("S")))/50.0)
             D[i][15]=b15
             b16=((float(s.count("T")))/50.0)
             D[i][16]=b16
             b17=((float(s.count("V")))/50.0)
             D[i][17]=b17
             b18=((float(s.count("W")))/50.0)
             D[i][18]=b18
             b19=((float(s.count("Y")))/50.0)
             D[i][19]=b19
             #a=[b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19]
             #print(str(b1))                  
             #print(D)
             frame=pd.DataFrame(D)
frame.to_csv('aac_piece.csv',header=None,index=None)
             #fseq2.write(str(b19)+"\n")

          
             #fseq2.writerows(a) #在1行1列写入bit  
           #在1行2列写入huang  
             #a.write(1,0,'xuan') #在2行1列写入xuan  
             #a.save('mini.xls')  #保存                      
             #fseq2.write(x)
#             a.append("item")
#             for i in a:
            
#                 fseq2.write("\n")
              
            
          
             
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
#             b=str((float(s.count("A")))/50)
             #b=str(b)
             #print str((b)+" "+(b1)+"\n")
#             s=[]
#             for i in range(20)
             #fseq2.writecell((b)+""+(b1)+"\n")
          
             #fseq2.write(str(s.count("A"))+"\n")
             #fseq2.write(((b)+" "+(b1)+"\n"))
             
