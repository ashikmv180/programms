#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


a=pd.read_csv(r"C:\Users\ASHIK M V\Downloads\datasets_7721_10938_Mall_Customers.csv")


# In[4]:


a.head(7)


# In[5]:


X=a.iloc[:,3:]


# In[6]:


X


# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


wcss= []


# In[11]:


a=range(1,10)


# In[13]:


for i in a:
    kme=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kme.fit(X)
    wcss.append(kme.inertia_)
    


# In[29]:


plt.plot(a,wcss)
plt.xlabel("k")
plt.ylabel("wcss")
plt.show()


# In[30]:


kmea=KMeans(n_clusters=5,init='k-means++',random_state=25)


# In[31]:


kmea.fit(X)


# In[48]:


y_pred=kmea.predict([[20,67]])


# In[49]:


y_pred


# In[50]:


y=kmea.fit_predict(np.asarray(X))


# In[51]:


y


# In[52]:


import pickle
file='finalizes model.pickle'
pickle.dump(kmea, open(file, 'wb'))


# In[53]:


load=pickle.load(open(file,'rb'))


# In[54]:


load.predict([[30,56]])


# In[ ]:




