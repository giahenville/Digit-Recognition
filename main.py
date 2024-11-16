import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 

# load dataset
mnist = tf.keras.datasets.mnist 
# split into training data and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()