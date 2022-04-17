#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


data = pd.read_csv("PS_20174392719_1491204439457_log.csv")
print(data.head())


# In[5]:


print(data.isnull().sum())


# In[6]:


print(data.type.value_counts())


# In[7]:


type = data["type"].value_counts()
transactions = type.index
quantity = type.values


# In[8]:


import plotly.express as px


# In[10]:


figure = px.pie(data,
               values = quantity,
               names = transactions, hole = 0.5,
               title = "Distribution of Transaction Type")
figure.show()


# In[14]:


#checking the correlation

correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending = False))


# In[17]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# In[24]:


#splitting the data

from sklearn.model_selection import train_test_split


# In[19]:


x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[25]:


#training the machine learning model

from sklearn.tree import DecisionTreeClassifier


# In[21]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[23]:


#prediction

#features = [type, amount, oldbalanceOrg, newbalanceOrig]


features = np.array([[4,9000.60, 9000.60, 0.0]])
print(model.predict(features))


# In[ ]:




