import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np


def get_dataset():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()


    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    

    x_train_reshaped = np.reshape(x_train,(60000,784))
    x_test_reshaped = np.reshape(x_test,(10000,784))


    x_mean = np.mean(x_train_reshaped)
    x_std = np.std(x_train_reshaped)

    epsilon = 1e-10

    x_train_norm = (x_train_reshaped - x_mean) / (x_std + epsilon)
    x_test_norm = (x_test_reshaped - x_mean) / (x_std + epsilon)

    return ((x_train_norm,y_train_encoded),(x_test_norm,y_test_encoded))


