import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from AI import get_dataset


checkpoint_path = "training_1/cp.pkt"

def create_model():
    model = Sequential([
        layers.InputLayer(input_shape = (28,28,1)),
        layers.RandomRotation(0.2),
        layers.Conv1D(64,3,activation="relu"),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dense(64,activation="relu"),
        layers.Dense(10,activation="softmax")
    ])

    model.compile(
        optimizer = 'sgd',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model

def train_model(model,x_train,y_train,x_test,y_test,epochs = 10):

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,save_weights_only=True,verbose = 1)
    model.fit(x_train,y_train,epochs = epochs,validation_data = (x_test,y_test),callbacks=[cp_callback])



def evaluate_model(model,x_test,y_test):
    loss , acc = model.evaluate(x_test,y_test,verbose = 2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))



if __name__ == '__main__':
    model = create_model()
    model.summary()
    (x_train,y_train),(x_test,y_test) = get_dataset.get_mnist_data(False)
    (x_train,y_train),(x_test,y_test) = get_dataset.process_image(x_train,x_test,y_train,y_test)
    
    train_model(model,x_train,y_train,x_test,y_test,epochs=20)
    evaluate_model(model,x_test,y_test)
