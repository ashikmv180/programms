#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[39]:


df = pd.read_csv(r"C:\Users\ASHIK M V\Downloads\spam.csv",encoding='ISO-8859-1',header=None,skiprows=1)


# In[40]:


df.head(7)
col=['label','message','Unnamed: 2','Unnamed: 3','Unnamed: 4']
df.columns=col


# In[41]:


df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)


# In[42]:


df


# In[43]:


df


# In[44]:


df.isnull().sum()


# In[45]:


len(df)


# In[57]:


df['label'].value_counts()


# In[60]:


ham = df[df['label']=='ham']


# In[61]:


spam = df[df['label']=='spam']


# In[62]:


spam


# In[68]:


ham.shape


# In[69]:


spam.shape


# In[67]:


data


# In[71]:


ham = ham.sample(spam.shape[0])


# In[73]:


ham.shape


# In[74]:


ham


# In[76]:


data = ham.append(spam,ignore_index=True)


# In[77]:


data.shape


# In[79]:


plt.hist(data[data['label']==ham], bins=100, alpha=0.7)


# In[80]:


plt.show()


# In[86]:


a=[]
for i  in data["message"]:
    a.append(len(i))
print(a)
    


# In[91]:


b = pd.DataFrame(a)


# In[92]:


a


# In[93]:


b


# In[105]:


b.columns=['length']


# In[106]:


b


# In[107]:


b


# In[111]:


data.shape


# In[112]:


b.shape


# In[116]:


data1= pd.concat([data,b],axis=1)


# In[119]:


plt.hist(data1[data1['label']=='ham']['length'], bins=100, alpha=0.7)
plt.hist(data1[data1['label']=='spam']['length'], bins=100, alpha=0.7)


# In[120]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[121]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[122]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[123]:


from sklearn.pipeline import Pipeline


# In[127]:


data1.head()


# In[150]:


X_train,X_test,y_train,y_test = train_test_split(data1['message'],data1['label'],test_size=0.3,random_state=0,shuffle=True,stratify=data1['label'])


# In[151]:


X_train


# In[172]:


y_train


# In[183]:


#vectorizer = TfidfVectorizer()


# In[184]:


#X_train = vectorizer.fit_transform(X_train)


# In[187]:


#X_train


# In[188]:


clfe = Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier(n_estimators=100,n_jobs=-1))])


# In[189]:


clfe.fit(X_train,y_train)


# In[190]:


y_pred = clfe.predict(X_test)


# In[191]:


y_pred


# In[180]:


accuracy_score(y_test,y_pred)


# In[181]:


clf.score(X_test,y_test)


# In[182]:


clf.predict(['congrats you have free ticket to usa,text 12445'])


# In[ ]:




