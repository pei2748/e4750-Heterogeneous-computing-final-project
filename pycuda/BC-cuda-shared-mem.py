import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle
import time

import math
from pycuda import gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit



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





data = pd.read_csv('data.csv')    # SVM only accepts numerical values. 
diagnosis_map = {'M':1, 'B':-1}
# drop last column (extra column added by pd)
data['diagnosis'] = data['diagnosis'].map(diagnosis_map)    
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
Y = data.loc[:, 'diagnosis']  # all rows of 'diagnosis' 
X = data.iloc[:, 1:]


X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)
X.insert(loc=len(X.columns), column='intercept', value=1)


# filter features
remove_correlated_features(X)
remove_less_significant_features(X, Y)


print("splitting dataset into train and test sets...")
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
    

############### train model with cuda ####################

length_of_features = X_train.to_numpy().shape[1]
num_examples = X_train.to_numpy().shape[0]
dim0 = 16 #32
dim1 = 64 #32
block_dim = (dim0, dim1, 1)
#grid_dim = (math.ceil(length_of_features/dim0), 1, 1)
grid_dim = (1, 1, 1)

arr = np.arange(num_examples)

reg_strength = 10000 # regularization strength
learning_rate = 0.000001
max_epochs = 200
prg_sgd = SourceModule(open("../kernels/sgd_bc_sh_mem.cu", "r").read())


func = prg_sgd.get_function("sgd")
weights = np.zeros(length_of_features).astype(np.float32)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

time_shuffle = 0
time_gpu = 0
s = time.time()

for i in range(0,200):
    _s = time.time()
    X_train, y_train = shuffle(X_train, y_train)
    #arr = shuffle(arr)
    _e = time.time()
    time_shuffle += (_e - _s) * 1e+3 # in ms
    # Transfer variables onto device
    _start = cuda.Event()
    _end = cuda.Event()
    _start.record()

    d_x = gpuarray.to_gpu(X_train.astype(np.float32));
    #d_index = gpuarray.to_gpu(arr.astype(np.float32));
    d_y = gpuarray.to_gpu(y_train.astype(np.float32));
    d_weights = gpuarray.to_gpu(weights);
    func(d_x, d_y, d_weights,
        np.float32(reg_strength),
        #np.int32(max_epochs),     
        np.float32(learning_rate),
        np.int32(num_examples),
       # d_index,
        block=block_dim,
        grid=grid_dim)
    _end.record()
    _end.synchronize()
    weights = d_weights.get()
    time_gpu +=   _start.time_till(_end) # in ms


    
e = time.time()
time = (e - s) # in s
time_total = (time_shuffle + time_gpu)*1e-3 # in s

print("total time : ", time_total)
print("time python : ", time)

W = weights
#print(W)

y_test_predicted = np.array([])


for i in range(X_test.shape[0]):
    yp = np.sign(np.dot(W, X_test.to_numpy()[i]))
    y_test_predicted = np.append(y_test_predicted, yp)


print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))

print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))

print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    



