# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:34:41 2021

@univariate cnn-lstm example
"""
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# define dataset
X = array([[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60], [40, 50, 60, 70]])
y = array([50, 60, 70, 80])
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
X = X.reshape((X.shape[0], 2, 2, 1))
# define model

def cnn():
    input_shape = (2, 2, 1)
    X_input = Input(input_shape)

    X=TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(2, 2, 1))(X_input)
    X=TimeDistributed(MaxPooling1D(pool_size=2))(X)
    X=TimeDistributed(Flatten())(X)
#    X = Dense(5, activation = 'softmax', name = 'fc' + str(5))(X)
    X=LSTM(50, activation='relu')(X)
    X=Dense(1)(X)

    model = Model(inputs = X_input, outputs = X, name = 'ResNet50')
    return model


model=cnn()
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)
x_input = array([50, 60, 70, 80])
x_input = x_input.reshape((1, 2, 2, 1))
yhat = model.predict(x_input, verbose=0)
print(yhat)