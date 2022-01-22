#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# You should build a machine learning pipeline using a linear regression model. In particular, you should do the following:
# - Load the `housing` dataset using [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html). You can find this dataset in the datasets folder.
# - Split the dataset into training and test sets using [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). 
# - Train and test a linear regression model using [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
# - Check the documentation to identify the most important hyperparameters, attributes, and methods of the model. Use them in practice.

# ## Importing libraries 
# 
# 

# In[23]:


import pandas as pd
import sklearn.model_selection 
import sklearn.linear_model


# ## Loading dataset

# In[9]:


df = pd.read_csv("../../datasets/housing.csv")
df = df.set_index("id")
df.head()


# ## Splitting 

# In[18]:


x = df.drop(["price"],axis=1)
y = df["price"]
x_train , x_test , y_train , y_test = sklearn.model_selection.train_test_split(x,y)

print("df:",df.shape)
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:" , y_test.shape)


# ## Training a model

# In[24]:


model = sklearn.linear_model.LinearRegression()
model.fit(x_train , y_train)


# ## Testing the trained Model

# In[25]:


y_predicted = model.predict(x_test)
y_predicted


# In[29]:


y_predicted = model.predict(x_test)
mse = sklearn.metrics.mean_squared_error(y_test, y_predicted)
mse


# In[ ]:





# In[ ]:




