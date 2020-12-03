#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# # Creating random numbers of nodes for radial structure

# In[2]:


# [[3],[no of fuzes ineach feeders],[no of transformers ineach fuze]]
feeders=[3]


# In[3]:


#number of feeders=sum(feeders) i.e 3
fuzes=np.random.randint(low=5, high=14, size=3)
print(fuzes)
print(sum(fuzes))


# In[4]:


#number of fuzes=sum(fuzes) i.e 17
transformers=np.random.randint(low=2, high=13, size=sum(fuzes))
print(transformers)
print(sum(transformers))


# In[5]:


#number of transformers=sum(transformers) i.e 92
customers=np.random.randint(low=2,high=19,size=sum(transformers))
print(customers)
print(sum(customers))


# In[6]:


#total number of nodes
totcols=1+sum(feeders)+sum(fuzes)+sum(transformers)+sum(customers)
print(totcols)


# In[7]:


ex=pd.DataFrame(columns=np.arange(0,totcols+1))


# In[8]:


ex.rename(columns={totcols:'y'},inplace=True)


# In[9]:


ex


# # Building the network

# In[10]:


for i in range(5000):
    row=[0 for k in range(totcols)]
    j=np.random.randint(0,totcols)
    row[j]=1
    row.append(j+1)
    ex.loc[i]=row
for i in range(5000,6000):
    row=[0 for k in range(totcols+1)]
    ex.loc[i]=row


# In[11]:


ex = ex.sample(frac=1).reset_index(drop=True)


# In[12]:


col=0


# In[13]:


ex['substation']=ex[0]


# In[14]:


for i in range(3):
    col+=1
    name='feeder '+ str(i+1)
    ex[name]=np.where(ex['substation']+ex[col]==1,1,0)


# In[15]:


ex


# In[16]:


#fuzes 
c=0 #Representing the node upstream from which the current node is driven i.e the feeder
fuzecount=0 #Representing the node value i.r the fuze
for i in fuzes:
    c+=1
    for j in range(i):
        col+=1
        fuzecount+=1
        name='fuze '+str(fuzecount)
        feedername='feeder '+str(c)
        ex[name]=np.where(ex[feedername]+ex[col]==1,1,0) #Will result in 1 if eitherthe current node or
                                                         #the node upstream is anomalous. 


# In[17]:


#transformers 
c=0
transcount=0
for i in transformers:
    c+=1
    for j in range(i):
        col+=1
        transcount+=1
        name='transformer '+str(transcount)
        fuzename='fuze '+str(c)
        ex[name]=np.where(ex[fuzename]+ex[col]==1,1,0)


# In[18]:


#sensors 
c=0
sensorcount=0
for i in customers:
    c+=1
    for j in range(i):
        col+=1
        sensorcount+=1
        name='sensor '+str(sensorcount)
        transname='transformer '+str(c)
        ex[name]=np.where(ex[transname]+ex[col]==1,1,0)


# In[19]:


ex.select_dtypes(include=['object'])


# In[20]:


ex.drop(ex.iloc[:,:totcols],axis=1, inplace=True)


# In[21]:


ex


# # Model training

# In[22]:


ex['substation']=ex['substation'].astype(int)
ex['y']=ex['y'].astype(int)
ex.info()


# In[23]:


dropped=1+sum(feeders)+sum(fuzes)+sum(transformers)


# In[24]:


features=ex.drop(ex.iloc[:,:dropped+1],axis=1)
output=ex['y']


# In[25]:


features.head()


# In[26]:


output.head()


# In[27]:


xtrain,xtest,ytrain,ytest=train_test_split(features,output,test_size=0.2,random_state=0)


# # Random forest

# In[28]:


rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)


# In[29]:


predicted=rf.predict(xtest)
pred1=rf.predict(xtrain)
print('accuracy_score on train dataset : ', accuracy_score(pred1,ytrain))
# Accuracy Score on test dataset
accuracy_test = accuracy_score(ytest,predicted)
print('accuracy_score on test dataset : ', accuracy_test)


# In[30]:


print(totcols)

