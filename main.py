import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

# load dataset
mnist = tf.keras.datasets.mnist 
# split into training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize (every value is between 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# neural network
model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax')) # output layer; all 10 neurons add up to 1

# compile model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])