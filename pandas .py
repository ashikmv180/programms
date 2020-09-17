#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
a = pd.read_table(r"http://bit.ly/chiporders")


# In[47]:


a.describe()


# In[23]:


b = pd.read_table("http://bit.ly/movieusers",sep="|",header=None,names=['userid','name','ashik'])


# In[48]:


b.dtypes


# In[31]:


c = b.userid


# In[32]:


d = b['userid']


# In[37]:


e= c +' ' +d


# In[39]:


e.head(5)


# In[44]:


b["asdfgh"] = e


# In[45]:


b


# In[46]:


b.describe()


# In[49]:


b.head()


# In[50]:


b.columns


# In[53]:


b.rename(columns={'userid':'UserId'})


# In[56]:


g.columns=['a','b','d']


# In[62]:


e  = ['a','b','c','f']


# In[63]:


b.columns=e


# In[ ]:





# In[64]:


b


# In[83]:


b.drop([1, 4], axis=0, inplace=True)


# In[82]:


b


# In[84]:


b


# In[85]:


b.a.values


# In[89]:


b.sort_values(['b','c'])


# In[88]:


b.f.order()


# In[92]:


a = pd.read_table("http://bit.ly/imdbratings",sep=',')


# In[93]:


a.head()


# In[95]:


a.shape


# In[97]:


for i in a.duration:
    print(i)


# In[98]:


boole=[]
for i in a.duration:
    if(i>=200):
        boole.append(True)
    else:
        boole.append(False)


# In[99]:


boole[5]


# In[101]:


for i in boole[2:56]:
    print(i)


# In[102]:


d= pd.Series(boole)


# In[104]:


d.head(250)


# In[107]:


a[d]


# In[108]:


b.head()


# In[122]:


pd.get_dummies(a.genre)


# In[112]:


a[a.duration>=200]['genre']


# In[119]:


a.loc(["genre"=567])


# In[121]:


pd.read_table(r"https://www.news18.com/cricketnext/profile/virat-kohli/batting-3993.html")


# In[123]:


import pandas as pd


# In[205]:


a=pd.read_csv(r"C:\Users\ASHIK M V\Downloads\datasets_21716_27925_50_Startups.csv")


# In[206]:


a.head(5)


# In[207]:


b=pd.get_dummies(a.State)


# In[208]:


b


# In[209]:


d=pd.concat([a,b],axis="columns")


# In[210]:


d


# In[211]:


e = d.drop(["State","Florida"],axis='columns')


# In[212]:


X = e.drop(["Profit"],axis="columns")


# In[213]:


y = d.Profit


# In[214]:


X


# In[215]:


y


# In[216]:


from sklearn.model_selection import train_test_split


# In[217]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25 )


# In[218]:


from sklearn.linear_model import LinearRegression
lin = LinearRegressionin()
lin.fit(X_train, y_train)


# In[219]:


from sklearn.linear_model import LinearRegression
lin = LinearRegressionin()


# In[220]:


from sklearn.linear_model import LinearRegression


# In[221]:


log = LinearRegression()
log.fit(X_train,y_train)


# In[222]:


lin.predict([[76524.01,99281.34,140574.81,0,1]])


# In[223]:


y_pred([[77044.01,99281.34,140574.81,0,1]])


# In[224]:


print(metrics.accuracy_score(y_test,y_pred))


# In[225]:


from sklearn import metrics


# In[226]:


from sklearn.metrics import r2_score


# In[227]:


r2_score(y_test,y_pred)


# In[ ]:


lin.predict(X)


# In[228]:


lin.predict([[151.51,101145.55,407934.54,0,0]])


# In[229]:


import pandas as pd


# In[230]:


a=pd.read_csv(r"C:\Users\ASHIK M V\Downloads\datasets_21716_27925_50_Startups.csv")


# In[231]:


a.head()


# In[250]:


X = a.iloc[-1,:].values


# In[251]:


X


# In[252]:


X = a.iloc[:,:].values


# In[264]:


X


# In[265]:


from sklearn.preprocessing import LabelEncoder


# In[266]:


lab = LabelEncoder()


# In[267]:


X[:,-2] = lab.fit_transform(X[:,-2])


# In[268]:


X


# In[269]:


X.head(3)


# In[273]:


a


# In[271]:


a = pd.DataFrame(X)


# In[272]:


a


# In[275]:


import pandas as pd
a = pd.read_csv(r"C:\Users\ASHIK M V\Downloads\datasets_122398_315766_full.csv")


# In[276]:


a.head(2)


# In[293]:


b=a.iloc[:,:].values


# In[294]:


b


# In[282]:


from sklearn.preprocessing import LabelEncoder


# In[283]:


lbb = LabelEncoder()


# In[292]:





# In[299]:


b[1,:]


# In[301]:


b[:,-10] = lbb.fit_transform(b[:,-10])


# In[303]:


a = pd.read_csv(r"C:\Users\ASHIK M V\Downloads\datasets_21716_27925_50_Startups.csv")


# In[304]:


a.head()


# In[306]:


b= a.iloc[:,:].values


# In[307]:


b[:,-1]


# In[308]:


b[:,-2] = lbb.fit_transform(b[:,-2])


# In[309]:


b


# In[310]:


c = pd.DataFrame(b)


# In[311]:


c.head()


# In[ ]:




