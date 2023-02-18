import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle 

def get_mnist_data(show_image = False):
    (x_train,y_train),(x_test,y_test) = mnist.load_data()

    if show_image:
        plt.imshow(x_train[0],cmap="binary")
        plt.show()

    return (x_train,y_train),(x_test,y_test)

def process_image(x_train,x_test,y_train,y_test):

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)
    

    x_train_reshaped = np.reshape(x_train,(60000,28,28,1))
    x_test_reshaped = np.reshape(x_test,(10000,28,28,1))

    x_train_scale = x_train_reshaped / 255
    x_test_scale =  x_test_reshaped / 255

    return ((x_train_scale,y_train_encoded),(x_test_scale,y_test_encoded))


if __name__ == "__main__":  
    (x_train,y_train),(x_test,y_test) = get_mnist_data(show_image=True)
    print(x_train.shape)
    #print((x_train[0]))
