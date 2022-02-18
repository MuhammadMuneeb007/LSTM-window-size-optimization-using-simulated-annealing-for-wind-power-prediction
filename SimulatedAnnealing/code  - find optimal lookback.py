#!/usr/bin/env python
# coding: utf-8

# In[11]:

from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot
from math import sqrt
from tabnanny import verbose
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import median_absolute_error
from tensorflow.keras.optimizers import RMSprop, SGD, Adam , Adadelta
from pandas import concat
from numpy import concatenate
from pandas import DataFrame
from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli 
from bitstring import BitArray
from sklearn.preprocessing import MinMaxScaler
from numpy import arange, sqrt, exp, pi 
import scipy.integrate as integral
from scipy import special
from scipy.special import erf
from math import e 
import math



# In[12]:


def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    # evaluate the initial point
    best_eval = objective(best)
    # current working solution
    curr, curr_eval = best, best_eval
    scores = list()
    # run the algorithm
    for i in range(n_iterations):
        # take a step
        candidate = curr + randn(len(bounds)) * step_size
        # evaluate candidate point
        candidate_eval = objective(candidate)
        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            scores.append(best_eval)
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))
        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval, scores]


# In[13]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[14]:
    
results = pd.DataFrame(columns=['a','b','c','d','e','f','g','h','i','j','k'])
# Windfarm 1

def train(x):
    lookback = int(x)
    print("Lookback Information",lookback)
    
    # Read the file
    dataset = read_csv('z1.csv', header=0, index_col=0)
    
    # The power value which we will predict
    y = dataset['final'].values
    del dataset['final']
    X = dataset.values
    
    # Convert data to timestep
    X = series_to_supervised(X,n_in=lookback).values
    y = y[lookback:]


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    # As we have 16 features so we selected only 16 features.

    X_train = X_train.reshape(X_train.shape[0],lookback+1,16)
    X_test = X_test.reshape(X_test.shape[0],lookback+1,16) 
    X_val = X_val.reshape(X_val.shape[0],lookback+1,16) 

    model = Sequential()
    model.add(LSTM(200, input_shape=(X_train.shape[1], 16),return_sequences=True))
    model.add(LSTM( 100,return_sequences=True))
    model.add(LSTM( 50, return_sequences=False))
    model.add(Dense(1))
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.008)
    model.compile(loss='mse', optimizer=rmsprop)
    model.fit(X_train, y_train, batch_size=512, epochs=30, verbose=0 ,validation_data=(X_val,y_val),shuffle=True ) 

    # Validation Set 
    yhat = model.predict(X_val)
    vmse = mean_squared_error(yhat,y_val)
    vmae = mean_absolute_error(yhat,y_val)
    vrmse = sqrt(mean_squared_error(yhat,y_val))
    vr2_Score = r2_score(yhat,y_val)
    vexplained_variance_Score = explained_variance_score(yhat,y_val)
    vrms = sqrt(mean_squared_error(yhat,y_val))
    print('Validation MSE: ', vmse)
    print('Validation MAE:', vmae)
    print('Validation RMSE: ',  vrms)
    print('Validation r2_score:',  vr2_Score)  #best if close to one
    print('Validation Explained_variance_score: ', vexplained_variance_Score)
    
    print("\n")
    # Test Set 
    yhat = model.predict(X_test)
    tmse = mean_squared_error(yhat,y_test)
    tmae = mean_absolute_error(yhat,y_test)
    trmse = sqrt(mean_squared_error(yhat,y_test))
    tr2_Score = r2_score(yhat,y_test)
    texplained_variance_Score = explained_variance_score(yhat,y_test)
    trms = sqrt(mean_squared_error(yhat,y_test))
    print('Test MSE: ', tmse)
    print('Test MAE:',  tmae)
    print('Test RMSE: ',  trms)
    print('Test r2_score:',  tr2_Score)  #best if close to one
    print('Test Explained_variance_score: ', texplained_variance_Score)
    print([vmse,vmae,vrmse,vr2_Score,vexplained_variance_Score,tmse,tmae,trmse,tr2_Score,texplained_variance_Score,lookback])
    results.loc[len(results)] = [vmse,vmae,vrmse,vr2_Score,vexplained_variance_Score,tmse,tmae,trmse,tr2_Score,texplained_variance_Score,lookback]
    results.to_csv("w11.txt",sep="\t")
    return vmse
# seed the pseudorandom number generator
 
 
seed(1)

# Input space for optimization
# Time step information. We considered 1 to 500 lookback information
from datetime import datetime
now = datetime.now()
bounds = asarray([[21, 501]])
# define the total iterations
n_iterations = 20
# define the maximum step size for simulated annealing
step_size = 10
# initial temperature
temp = 10
# perform the simulated annealing search
best, score, scores = simulated_annealing(train, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))
# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()

later = datetime.now()
difference = ((later-now).total_seconds())
print(difference)
# In[ ]:

results = pd.DataFrame(columns=['a','b','c','d','e','f','g','h','i','j','k'])
# Windfarm 2
def train(x):
    lookback = int(x)
    print("Lookback Information",lookback)
    
    # Read the file
    dataset = read_csv('z2.csv', header=0, index_col=0)
    
    # The power value which we will predict
    y = dataset['final'].values
    del dataset['final']
    X = dataset.values
    
    # Convert data to timestep
    X = series_to_supervised(X,n_in=lookback).values
    y = y[lookback:]


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    # As we have 16 features so we selected only 16 features.

    X_train = X_train.reshape(X_train.shape[0],lookback+1,16)
    X_test = X_test.reshape(X_test.shape[0],lookback+1,16) 
    X_val = X_val.reshape(X_val.shape[0],lookback+1,16) 

    model = Sequential()
    model.add(LSTM(200, input_shape=(X_train.shape[1], 16),return_sequences=True))
    model.add(LSTM( 100,return_sequences=True))
    model.add(LSTM( 50, return_sequences=False))
    model.add(Dense(1))
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.008)
    model.compile(loss='mse', optimizer=rmsprop)
    model.fit(X_train, y_train, batch_size=512, epochs=30, verbose=0 ,validation_data=(X_val,y_val),shuffle=True ) 

    # Validation Set 
    yhat = model.predict(X_val)
    vmse = mean_squared_error(yhat,y_val)
    vmae = mean_absolute_error(yhat,y_val)
    vrmse = sqrt(mean_squared_error(yhat,y_val))
    vr2_Score = r2_score(yhat,y_val)
    vexplained_variance_Score = explained_variance_score(yhat,y_val)
    vrms = sqrt(mean_squared_error(yhat,y_val))
    print('Validation MSE: ', vmse)
    print('Validation MAE:', vmae)
    print('Validation RMSE: ',  vrms)
    print('Validation r2_score:',  vr2_Score)  #best if close to one
    print('Validation Explained_variance_score: ', vexplained_variance_Score)
    
    print("\n")
    # Test Set 
    yhat = model.predict(X_test)
    tmse = mean_squared_error(yhat,y_test)
    tmae = mean_absolute_error(yhat,y_test)
    trmse = sqrt(mean_squared_error(yhat,y_test))
    tr2_Score = r2_score(yhat,y_test)
    texplained_variance_Score = explained_variance_score(yhat,y_test)
    trms = sqrt(mean_squared_error(yhat,y_test))
    print('Test MSE: ', tmse)
    print('Test MAE:',  tmae)
    print('Test RMSE: ',  trms)
    print('Test r2_score:',  tr2_Score)  #best if close to one
    print('Test Explained_variance_score: ', texplained_variance_Score)
    results.loc[len(results)] = [vmse,vmae,vrmse,vr2_Score,vexplained_variance_Score,tmse,tmae,trmse,tr2_Score,texplained_variance_Score,lookback]
    results.to_csv("w22.txt",sep="\t")
    return vmse
# seed the pseudorandom number generator
seed(1)

# Input space for optimization
# Time step information. We considered 1 to 500 lookback information
from datetime import datetime
now = datetime.now()
bounds = asarray([[21, 501]])
# define the total iterations
n_iterations = 20
# define the maximum step size for simulated annealing
step_size = 10
# initial temperature
temp = 10
# perform the simulated annealing search
best, score, scores = simulated_annealing(train, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))
# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
later = datetime.now()
difference = ((later-now).total_seconds())
print(difference)
# In[ ]:
results = pd.DataFrame(columns=['a','b','c','d','e','f','g','h','i','j','k'])
# Windfarm 3
def train(x):
    lookback = int(x)
    print("Lookback Information",lookback)
    
    # Read the file
    dataset = read_csv('z3.csv', header=0, index_col=0)
    
    # The power value which we will predict
    y = dataset['final'].values
    del dataset['final']
    X = dataset.values
    
    # Convert data to timestep
    X = series_to_supervised(X,n_in=lookback).values
    y = y[lookback:]


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    # As we have 16 features so we selected only 16 features.

    X_train = X_train.reshape(X_train.shape[0],lookback+1,16)
    X_test = X_test.reshape(X_test.shape[0],lookback+1,16) 
    X_val = X_val.reshape(X_val.shape[0],lookback+1,16) 

    model = Sequential()
    model.add(LSTM(200, input_shape=(X_train.shape[1], 16),return_sequences=True))
    model.add(LSTM( 100,return_sequences=True))
    model.add(LSTM( 50, return_sequences=False))
    model.add(Dense(1))
    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.008)
    model.compile(loss='mse', optimizer=rmsprop)
    model.fit(X_train, y_train, batch_size=512, epochs=30, verbose=0 ,validation_data=(X_val,y_val),shuffle=True ) 

    # Validation Set 
    yhat = model.predict(X_val)
    vmse = mean_squared_error(yhat,y_val)
    vmae = mean_absolute_error(yhat,y_val)
    vrmse = sqrt(mean_squared_error(yhat,y_val))
    vr2_Score = r2_score(yhat,y_val)
    vexplained_variance_Score = explained_variance_score(yhat,y_val)
    vrms = sqrt(mean_squared_error(yhat,y_val))
    print('Validation MSE: ', vmse)
    print('Validation MAE:', vmae)
    print('Validation RMSE: ',  vrms)
    print('Validation r2_score:',  vr2_Score)  #best if close to one
    print('Validation Explained_variance_score: ', vexplained_variance_Score)
    
    print("\n")
    # Test Set 
    yhat = model.predict(X_test)
    tmse = mean_squared_error(yhat,y_test)
    tmae = mean_absolute_error(yhat,y_test)
    trmse = sqrt(mean_squared_error(yhat,y_test))
    tr2_Score = r2_score(yhat,y_test)
    texplained_variance_Score = explained_variance_score(yhat,y_test)
    trms = sqrt(mean_squared_error(yhat,y_test))
    print('Test MSE: ', tmse)
    print('Test MAE:',  tmae)
    print('Test RMSE: ',  trms)
    print('Test r2_score:',  tr2_Score)  #best if close to one
    print('Test Explained_variance_score: ', texplained_variance_Score)
    results.loc[len(results)] = [vmse,vmae,vrmse,vr2_Score,vexplained_variance_Score,tmse,tmae,trmse,tr2_Score,texplained_variance_Score,lookback]
    results.to_csv("w33.txt",sep="\t")
    return vmse
# seed the pseudorandom number generator
seed(1)

# Input space for optimization
# Time step information. We considered 1 to 500 lookback information
from datetime import datetime
now = datetime.now()
bounds = asarray([[21, 501]])
# define the total iterations
n_iterations = 20
# define the maximum step size for simulated annealing
step_size = 10
# initial temperature
temp = 10
# perform the simulated annealing search
best, score, scores = simulated_annealing(train, bounds, n_iterations, step_size, temp)
print('Done!')
print('f(%s) = %f' % (best, score))
# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()
pyplot.save()
later = datetime.now()
difference = ((later-now).total_seconds())
print(difference)
# In[ ]:




