from AI.create_model import create_model
from AI.get_dataset import get_dataset

import tensorflow as tf
import os

checkpoint_path = "AI/training_1/cp.pkt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,save_weights_only=True,verbose = 1)


if __name__ == "__main__":
    model = create_model()
    (x_train,y_train),(x_test,y_test) = get_dataset()
    model.fit(x_train,y_train,epochs = 10,validation_data = (x_test,y_test),callbacks=[cp_callback])

