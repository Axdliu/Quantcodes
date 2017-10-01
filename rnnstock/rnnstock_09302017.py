#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:17:23 2017

@author: arnoleu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

# import data: S&P 500 index 
data = pd.read_csv('GSPC.csv', header=0, index_col=0)
data = data.set_index(pd.to_datetime(data.index))
start_date = dt.datetime(2017, 4, 1)
end_date =  dt.datetime(2017, 9, 30)
data_select = data[(data.index >= start_date) & (data.index <= end_date)]

# split into training and tesing data sets
split_date =  dt.datetime(2017, 8, 31)
training_set = data_select[data_select.index <= split_date]
test_set = data_select[data_select.index > split_date]

#  using [:,1:2] rather than [:,1] to make sure it is a matrix
training_set = training_set.iloc[:,4:5].values

# Feature Scaling
# this used normalization: xnorm = (x-min(x))/(max(x)-min(x))
# standardization: xstd = (x-mean(x))/std(x)
# this is different from using returns, but let's try it first
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:-1]
y_train = training_set[1:]
days_train = X_train.shape[0]

# Reshaping: before, it is a 2-D arrays. To change it into a 3-D array to show the timestep
# check 3D tensor with shape
X_train = np.reshape(X_train, (days_train, 1, 1))

# Building the RNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()
# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Making the predictions and visualising the results
# Getting the real stock price of 2017
real_stock_price = test_set.iloc[:,4:5].values
# Getting the predicted stock price of 2017
inputs = real_stock_price[0:-1]
inputs = sc.transform(inputs)
days_test = inputs.shape[0]
inputs = np.reshape(inputs, (days_test, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
actual_price = real_stock_price[1:]

# model evaluation
c = np.corrcoef(predicted_stock_price.T, y=actual_price.T)

# Visualising the results
plt.plot(test_set.index[1:], actual_price, color = 'red', label = 'Actual S&P 500 Index')
plt.plot(test_set.index[1:], predicted_stock_price, color = 'blue', label = 'Predicted S&P 500 Index')
plt.title('S&P 500 Index Prediction (Trained with normalized data')
plt.xlabel('Time')
plt.ylabel('S&P 500 Index')
plt.legend()
plt.show()


'''
Using Returns here
'''

training_set = data_select[data_select.index <= split_date]
test_set = data_select[data_select.index > split_date]
#  using [:,1:2] rather than [:,1] to make sure it is a matrix
# Feature Scaling using returns
training_set = training_set.iloc[:,4:5].pct_change().values

# Getting the inputs and the ouputs
X_train = training_set[1:-1]
y_train = training_set[2:]
days_train = X_train.shape[0]

# Reshaping: before, it is a 2-D arrays. To change it into a 3-D array to show the timestep
# check 3D tensor with shape
X_train = np.reshape(X_train, (days_train, 1, 1))

# Initialising the RNN
regressor = Sequential()
# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Making the predictions and visualising the results
# Getting the real stock price of 2017
real_stock_price = test_set.iloc[:,4:5].pct_change().values
# Getting the predicted stock price of 2017
inputs = real_stock_price[1:-1]
inputs = sc.transform(inputs)
days_test = inputs.shape[0]
inputs = np.reshape(inputs, (days_test, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = ((predicted_stock_price + 1).cumprod())*test_set.iloc[0,4]
actual_price = test_set.iloc[2:,4:5].values

# model evaluation
c = np.corrcoef(predicted_stock_price.T, y=actual_price.T)

# Visualising the results
plt.plot(test_set.index[2:], actual_price, color = 'red', label = 'Actual S&P 500 Index')
plt.plot(test_set.index[2:], predicted_stock_price, color = 'blue', label = 'Predicted S&P 500 Index')
plt.title('S&P 500 Index Prediction (Trained with returns)')
plt.xlabel('Time')
plt.ylabel('S&P 500 Index')
plt.legend()
plt.show()