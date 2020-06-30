#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np 
import statistics as st 
import random

df=pd.read_csv("MLDataMode.csv")
x=np.array(df)
def compute(xtrain,ytrain,xtest,ytest):
    d=[]
    for i in range(15):
        d.append([0,0,0,0])  #yes democrat , no democrat, yes rep , no rep
    Pd=0
    Pr=0
    for i in ytrain:
        if i == 'democrat':
            Pd=Pd+1   #total number of democrats
        else:
            Pr=Pr+1  # total number of republicans
    Pdem=Pd/(Pr+Pd)
    Prep=1-Pdem
    #print(d)
    for i in range(15):
        for j in range(348):
            if xtrain[j][i]=='y' and ytrain[j]=='democrat':
                d[i][0]=d[i][0]+1
            elif xtrain[j][i]=='n' and ytrain[j]=='democrat':
                d[i][1]=d[i][1]+1
            elif xtrain[j][i]=='y' and ytrain[j]=='republican':
                d[i][2]=d[i][2]+1
            elif xtrain[j][i]=='n' and ytrain[j]=='republican':
                d[i][3]=d[i][3]+1  
    confusion_matrix=[0,0,0,0] #TP,FN,TN,FN
    #testing
    count=0
    argmax=0
    for i in xtest:
        P1=Pdem
        P2=Prep
        for j in range(len(i)):
            if i[j]=='y':
                P1=P1*(d[j][0] / Pd)
                P2=P2*(d[j][2] / Pr)
            else:
                P1=P1*(d[j][1]/ Pd)
                P2=P2*(d[j][3]/ Pr)
        if(P1 > P2):
            if ytest[count]=='democrat':
                argmax=argmax+1
                confusion_matrix[0]+=1 #TP
            else:
                confusion_matrix[1]+=1  #FN
        else:
            if ytest[count]=='republican':
                argmax=argmax+1  #TN
                confusion_matrix[2]+=1
            else:
                confusion_matrix[3]+=1 #FP
        count=count+1
    #print(argmax/87)
    #confusion_matrix
    Accuracy= (confusion_matrix[0]+confusion_matrix[2])/(confusion_matrix[0]+confusion_matrix[1]+confusion_matrix[2]+confusion_matrix[3])
    Precision = confusion_matrix[0] /(confusion_matrix[0]+confusion_matrix[3])
    Recall=(confusion_matrix[0])/(confusion_matrix[0]+confusion_matrix[1]) 
    FScore=2*(Recall * Precision)/(Recall + Precision)
    #print(Accuracy)
    return Accuracy


demyes=0
demno=0
repyes=0
repno=0
def max1(a,b):
    if a > b:
        return 'y'
    else: 
        return 'n'
for j in range(16):
    for i in range(435):
        if x[i][j]=='n' and x[i][16]=='democrat':
            demno=demno+1
        elif x[i][j]=='y' and x[i][16]=='democrat':
            demyes=demyes+1 
        elif x[i][j]=='n' and x[i][16]=='republican':
            repno=repno+1
        elif x[i][j]=='y' and x[i][16]=='republican':
            repyes=repyes+1
    dem=max1(demyes,demno)
    rep=max1(repyes,repno)
    for i in range(435):
        if x[i][j]=='?' and x[i][16]=='democrat':
            x[i][j]=dem
        elif x[i][j]=='?' and x[i][16]=='republican':
            x[i][j]=rep

k=5
random.shuffle(x)
Y=x[:,16]
X=x[:,1:-1]
Avg=0
X=X.tolist()
Y=Y.tolist()
xtrain1=X[:348]  #train:1234 test: 5
ytrain1=Y[:348]
xtest1=X[348:]
ytest1=Y[348:]
Avg= compute(xtrain1,ytrain1,xtest1,ytest1)

#train: 1235 #test:4
xtrain2=X[:261] + X[348:]
ytrain2=Y[:261] + Y[348:]
xtest2=X[261:348]
ytest2=Y[261:348]
Avg=Avg+compute(xtrain2,ytrain2,xtest2,ytest2)

#test: 3
xtrain3=X[:174] + X[261:]
ytrain3=Y[:174] + Y[261:]
xtest3=X[174:261]
ytest3=Y[174:261]
Avg=Avg+compute(xtrain3,ytrain3,xtest3,ytest3)

#test: 2
xtrain4=X[:87] + X[174:]
ytrain4=Y[:87] + Y[174:]
xtest4=X[87:174]
ytest4=Y[87:174]
Avg=Avg+compute(xtrain4,ytrain4,xtest4,ytest4)
#test 1
xtrain5=X[87:]
ytrain5=Y[87:] 
xtest5=X[:87]
ytest5=Y[:87]
Avg=Avg+compute(xtrain5,ytrain5,xtest5,ytest5)

Avg=Avg/5
print(Avg)





                                   



# In[ ]:





# In[ ]:




