#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle
import cv2 as cv
import time


# In[2]:


#def remove_correlated_features(X):
#def remove_less_significant_features(X, Y):# >> MODEL TRAINING << #
#def compute_cost(W, X, Y):
#def calculate_cost_gradient(W, X_batch, Y_batch):
#def sgd(features, outputs):def init():


# In[3]:


# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped

def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped


# In[4]:


# set hyper-parameters and call init
# hyper-parameters are normally tuned using cross-validation
# but following work good enough
reg_strength = 10000 # regularization strength
learning_rate = 0.000001


# In[16]:


data = pd.read_csv('data.csv')    # SVM only accepts numerical values. 
# Therefore, we will transform the categories M and B into values 1 and -1 (or -1 and 1), respectively.
diagnosis_map = {'M':1, 'B':-1}
data['diagnosis'] = data['diagnosis'].map(diagnosis_map)    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
Y = data.loc[:, 'diagnosis']  # all rows of 'diagnosis' 
X = data.iloc[:, 1:]  # all rows of column 1 and ahead (features)# normalize the features using MinMaxScalar from
    # sklearn.preprocessing
X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)
    # first insert 1 in every row for intercept b
X.insert(loc=len(X.columns), column='intercept', value=1)
    


# In[17]:


# filter features
remove_correlated_features(X)
remove_less_significant_features(X, Y)


# In[18]:


# test_size is the portion of data that will go into test set
# random_state is the seed used by the random number generator
print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

 


# In[19]:


X_train = np.matrix(np.array(X_train).astype('float32'))


# In[20]:


X_test = np.matrix(np.array(X_test).astype('float32'))


# In[26]:


y_train = np.array(y_train)


# In[27]:


svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
## [init]
## [train]
start = time.time()
svm.train(X_train, cv.ml.ROW_SAMPLE, y_train)
end = time.time()


# In[28]:


print(end - start)


# In[29]:


y_test_predicted = np.array([])
s = time.time()
for i in range(X_test.shape[0]):
    yp = svm.predict(X_test[i])
    y_test_predicted = np.append(y_test_predicted, yp[1])

e = time.time()

print(e - s)


# In[30]:





# In[31]:


print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    
    


# In[ ]:





# In[ ]:





# In[ ]:




