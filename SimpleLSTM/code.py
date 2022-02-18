#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


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

difference = []
# In[3]:
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
for windfarm in ['z1','z2','z3']:
    for loop in range(1,500):
        lookback = loop
        print("Lookback Information",lookback)
        
        # Read the file
        dataset = read_csv(windfarm+ '.csv', header=0, index_col=0)
        
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
        mse = mean_squared_error(yhat,y_val)
        mae = mean_absolute_error(yhat,y_val)
        rmse = sqrt(mean_squared_error(yhat,y_val))
        r2_Score = r2_score(yhat,y_val)
        explained_variance_Score = explained_variance_score(yhat,y_val)
        rms = sqrt(mean_squared_error(yhat,y_val))
        print('Validation MSE: ', mse)
        print('Validation MAE:',  mae)
        print('Validation RMSE: ',  rms)
        print('Validation r2_score:',  r2_Score)  #best if close to one
        print('Validation Explained_variance_score: ', explained_variance_Score)
        
        print("\n")
        # Test Set 
        yhat = model.predict(X_test)
        mse = mean_squared_error(yhat,y_test)
        mae = mean_absolute_error(yhat,y_test)
        rmse = sqrt(mean_squared_error(yhat,y_test))
        r2_Score = r2_score(yhat,y_test)
        explained_variance_Score = explained_variance_score(yhat,y_test)
        rms = sqrt(mean_squared_error(yhat,y_test))
        print('Test MSE: ', mse)
        print('Test MAE:',  mae)
        print('Test RMSE: ',  rms)
        print('Test r2_score:',  r2_Score)  #best if close to one
        print('Test Explained_variance_score: ', explained_variance_Score)
        later = datetime.now()
        difference.append((later-now).total_seconds())
        print("Lookback : ",lookback," Time : ",difference)
    plt.plot(difference)
    plt.show()
    difference= []
plt.plot(difference)
plt.show()
