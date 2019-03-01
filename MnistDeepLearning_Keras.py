from __future__ import print_function
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import optimizers
from keras.utils import np_utils

np.random.seed(16)
epoch = 60
batchSize = 128
verbose = 1
classes = 10 
optm = optimizers.Adam() #Can be changed
hidden = 128
vSplit = 0.2 
dropout = 0.26

(XTrain, yTrain), (XTest, yTest) = mnist.load_data()
RESHAPED = 784
XTrain = XTrain.reshape(60000, RESHAPED)
XTest = XTest.reshape(10000, RESHAPED)
XTrain = XTrain.astype('float32')
XTest = XTest.astype('float32')
XTrain /= 255 #normalizing
XTest /= 255
print(XTrain.shape[0], 'train samples')
print(XTest.shape[0], 'test samples')

# convert class vectors to binary class matrices
YTrain = np_utils.to_categorical(yTrain, classes)
YTest = np_utils.to_categorical(yTest, classes)

model = Sequential()
model.add(Dense(hidden, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden//2))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer = optm,metrics=['accuracy'])
history = model.fit(XTrain, YTrain,batch_size=batchSize, epochs=epoch,verbose=verbose, validation_split=vSplit)

score = model.evaluate(XTest, YTest, batch_size = batchSize)
print("Test accuracy:", score[1])
