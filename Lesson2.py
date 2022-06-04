#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ## 2.1 Data splitting

# In[2]:


df=pd.read_csv('result_data.csv')


# Let's split the data set in the way recommended according to the `Sklearn` documentation - `30 by 70`. As presented in the description, such a sample is optimal, since the absolute majority of data must be found when training the model in order to obtain the most optimized model from the side of its accuracy
# 
# ### Stratification
# When dividing, we stratify the data to get the same percentage of the sample, so that there is no preponderance for any one class and such a situation does not affect incorrect training of the model

# In[3]:


df


# In[4]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[5]:


df=df[df['Rt']<5].reset_index(drop=True)


# ### Danger definition

# In[6]:


df1=df[df['Rt']<=0.7]
df1['Danger']=0


# In[7]:


df2=df[(df['Rt']>0.7) & (df['Rt']<=0.95)]
df2['Danger']=1


# In[8]:


df3=df[df['Rt']>0.95]
df3['Danger']=2


# In[9]:


df=pd.concat([df1, df2, df3]).reset_index(drop=True)


# In[10]:


X=df[['new_cases', 'new_deaths', 'Rt']]
y=df['Danger']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# ## 2.3 Classification
# 
# 
# ### KNeighborsClassifier
# ### RandomForestClassifier
# ### GaussianNB
# 
# ## Metrics
# 
# ### accuracy f1-score
# ### macro avg f1-score

# ## 2.4 Training

# In[12]:


from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[13]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
preds=neigh.predict(X_test)
print(classification_report(preds, y_test))


# In[14]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_preds=rfc.predict(X_test)
print(classification_report(rfc_preds, y_test))


# In[15]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_preds=gnb.predict(X_test)
print(classification_report(gnb_preds, y_test))


# ### Output
# The most optimal model will be the `KNeighborsClassifier` with accuracy f1-score = `0.78` and macro arg f1-score = `0.74`, since it showed the best result compared to others. `Random Forest Classifier` will not be taken because it has an explicit retraining

# ## Report
# * 2.1 Splitting the data set - the data set is divided into training and test samples
# * 2.3 Classification - 3 classification algorithms are selected
# * 2.4 Training - classification by hazard level has been made

# In[16]:


df.to_csv('result_data.csv', encoding='utf-8-sig', index=False)

