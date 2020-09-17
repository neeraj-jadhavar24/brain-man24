#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# #### Load Data

# In[2]:


data = pd.read_excel(r"C:\Users\NEERAJ\MFDS2.xlsx")


# In[3]:


data.head()


# In[4]:


encode = {'Test': {'Pass':1, 'Fail':0}}


# In[5]:


data.replace(encode,inplace=True)


# # 1) Data Statistics

# In[6]:


data.head()


# In[7]:


data.info()


# In[8]:


data.describe()


# In[9]:


data['Temperature'].plot(kind = 'bar')


# In[10]:


sns.pairplot(data)      


# In[11]:


data['Test'].value_counts()


# In[12]:


sns.countplot(data['Test'])


# In[13]:


data1 = data


# In[14]:


data1


# In[15]:


data1 = data1.rename(columns={"Inlet reactant concentration":"Inlet reactant conc"})


# In[16]:


data1.corr()


# In[17]:


corrmat = data1.corr()

sns.heatmap(corrmat, vmax=.8, square=True)


# ###### Temperature

# In[18]:


sns.distplot(data['Temperature']);


# In[19]:


data['Temperature'].skew()


# In[20]:


data['Temperature'].kurtosis()


# ##### Pressure

# In[21]:


sns.distplot(data['Pressure']);


# In[22]:


data['Pressure'].skew()


# In[23]:


data['Pressure'].kurtosis()


# ##### Feed Flow rate

# In[24]:


sns.distplot(data['Feed Flow rate']);


# In[25]:


data['Feed Flow rate'].skew()


# In[26]:


data['Feed Flow rate'].kurtosis()


# ##### Coolant Flow rate

# In[27]:


sns.distplot(data['Coolant Flow rate']);


# In[28]:


data['Coolant Flow rate'].skew()


# In[29]:


data['Coolant Flow rate'].kurtosis()


# ##### Inlet reactant concentraion

# In[30]:


sns.distplot(data['Inlet reactant concentration'])


# In[31]:


data['Inlet reactant concentration'].skew()


# In[32]:


data['Inlet reactant concentration'].kurtosis()


# # 2) Split Data 

# #### Dividing data into features and labels

# In[33]:


X = data.drop(['Test'],axis=1)
y = data['Test']


# #### Split dataset using split function from numpy library with 30% test data

# In[34]:


[X_train, X_test] = np.split(X, [int(0.7*X.shape[0])], axis=0)
[y_train, y_test] = np.split(y, [int(0.7*y.shape[0])], axis=0)


# In[35]:


y_train.value_counts()


# # 3) Logistic Regression Model

# #### Feature scaling

# Having features on a similar scale can help the gradient descent converge more quickly towards the minima.
# 

# Using standardization method

# In[36]:


X_train = (X_train - X_train.mean())/X_train.std()


# In[37]:


X_train.head()


# #### Adding bias term

# In[38]:


X_train.insert(0, 'bias', 1)    
X_train.head()


# #### Sigmoid function

# In[39]:


def sigmoid(z):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-z))


# #### Cost Function

# In[40]:


def cost(probability, label):
    # Computes the cost function for all the given predictions and labels
    return (-label*np.log(probability)-(1-label)*np.log(1-probability)).mean()


# #### Gradient

# In[41]:


def gradient(x,theta, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    h = sigmoid(np.dot(x, theta))
    return (1 / m) * np.dot(x.T, h-y)


# #### Gradient descent

# In[42]:


def gradient_descent(x, y, learning_rate, number_of_iterations):
    # performs iterations of gradient descent and returns parameters given features, labels, learning rate and number of iterations.
    theta = np.random.normal(0, 1, features.shape[1])
    for i in range(number_of_iterations):
        grad = gradient(x, theta, labels)
        theta -= learning_rate*grad
    return theta


# #### Predict 

# In[43]:


def predict(x, theta, threshold):
    #predict function that returns the predicted labels given features, parameters and threshold.
    return sigmoid(np.dot(x, theta)) >= threshold


# #### Misclassification

# In[44]:


def misclassification_error(predictions, labels):
    return (predictions!=labels).mean()


# #### Transforming test dataset

# In[45]:


X_test = (X_test - X_test.mean())/X_test.std()
X_test.insert(0,'bias',1)


# #### Initializing parameters

# In[46]:


theta = np.zeros(6)
theta_i = np.zeros(6)

alpha = 0.1             #learning_rate
test_cost = 1000        #cost function calculated on test samples


# #### Finalizing parameters considering training and test costs

# In[47]:


i=1
while 1:
    theta_i -= alpha*gradient(X_train, theta, y_train)               #Parameters are changed after every iteration
    train_cost = cost(sigmoid(np.dot(X_train, theta_i)), y_train)    #Training cost calculated
    temp_cost = cost(sigmoid(np.dot(X_test, theta_i)), y_test)       #Temporary test cost calculated 
    if temp_cost > test_cost:
        if alpha > 0.000001:
            alpha = alpha/10                                         #Learning rate readjusted after every iteration
            continue
        else:
            break
    test_cost = temp_cost
    theta = theta_i
    print(i, train_cost, test_cost)
    i = i + 1


# #### Theta

# In[48]:


theta


# #### Error in training dataset

# In[49]:


misclassification_error(predict(X_train, theta, 0.5), y_train)


# #### Error in test dataset

# In[50]:


misclassification_error(predict(X_test, theta, 0.5), y_test)


# # 4) Confusion Matrix

# In[51]:


def confusion_matrix(predictions, labels):
    true_positives = np.logical_and(predictions, labels).sum()   # 1 only when both prediction and label are 1
    false_positives = np.logical_and(predictions, np.logical_not(labels)).sum()  # 1 only when prediction is 1 and label is 0
    false_negatives = np.logical_and(np.logical_not(predictions), labels).sum()  # 1 only when prediction is 0 and label is 1
    true_negatives = np.logical_not(np.logical_or(predictions, labels)).sum()    # 1 only when both predictions and label are 0
    return pd.DataFrame(data={'Predicted 0': [true_negatives, false_negatives], 'Predicted 1': [false_positives, true_positives]}, index=['Actual 0', 'Actual 1'])


# In[52]:


X_all = (X-X.mean())/X.std()
X_all.insert(0, 'bias', 1)


# In[53]:


predictions = predict(X_all,theta,0.5)


# In[54]:


confusion_matrix(predictions,y)


# In the above table Class 0 : 'Fail' and Class 1 : 'Pass'

# #### Recall

# In[55]:


def recall(confusion_matrix):
    return (confusion_matrix['Predicted 0']['Actual 0'])/(confusion_matrix['Predicted 0']['Actual 0']+confusion_matrix['Predicted 1']['Actual 0'])


# In[56]:


recall(confusion_matrix(predictions,y))


# #### Precision

# In[57]:


def precision(confusion_matrix):
    return (confusion_matrix['Predicted 0']['Actual 0'])/(confusion_matrix['Predicted 0']['Actual 0']+confusion_matrix['Predicted 0']['Actual 1'])


# In[58]:


precision(confusion_matrix(predictions,y))


# #### F1-score

# In[60]:


def f1_score(confusion_matrix):
    return (2*confusion_matrix["Predicted 1"]["Actual 1"])/(2*confusion_matrix["Predicted 1"]["Actual 1"]+confusion_matrix["Predicted 0"]["Actual 1"]+confusion_matrix["Predicted 1"]["Actual 0"])


# In[61]:


f1_score(confusion_matrix(predictions,y))

