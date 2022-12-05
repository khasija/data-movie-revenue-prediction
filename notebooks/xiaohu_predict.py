#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[46]:


import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from python_files.data import GetData
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression


# In[5]:


data = GetData().get_data()['AllDataMerged_updated']
data.head()


# In[6]:


df = data.copy()


# In[7]:


df.shape


# In[8]:


# Apply log feature to budget and revnue

df['budget_log'] = np.log(df['budget'])
df['revenue_log'] = np.log(df['revenue'])


# In[16]:


# Define X and Y

X = df[['budget_log','runtime','production_companies_number','production_countries_number',
       'spoken_languages_number','director_number','producer_number','actor_number']]
y = df['revenue_log']


# In[38]:


y.head()


# In[17]:


# Split data
X_tain, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[18]:


X_test.shape


# In[19]:


# Split test data
X_test, X_pred, y_test, y_pred = train_test_split(X_test, y_test, test_size=0.05, random_state=1)


# In[20]:


X_pred.shape


# In[27]:


# Build pipeline

pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='mean')),
    ('robust_scaler', RobustScaler())])
pipeline


# In[28]:


# Fit and train

X_train_transformed = pipeline.fit_transform(X_tain)
X_test_trainsformed = pipeline.transform(X_test)


# In[33]:


pd.DataFrame(X_test_trainsformed)


# In[35]:


# Instantiate and train model

lin_model = LinearRegression()
lin_model.fit(X_train_transformed,y_train)


# In[36]:


# Score the model
lin_model.score(X_test_trainsformed,y_test)


# In[45]:


# Predict model
y_pred_log = lin_model.predict(X_pred)
pd.DataFrame(y_pred_log).head()


# In[44]:


# Convert log value to normal
y_pred = np.exp(y_pred_log)
pd.DataFrame(y_pred).head()

