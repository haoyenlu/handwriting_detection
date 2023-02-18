from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model():
    model = Sequential([
        Dense(128,activation="relu",input_shape=(784,)),
        Dense(128,activation="relu"),
        Dense(10,activation="softmax")
    ])

    model.compile(
        optimizer = 'sgd',
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    return model


if __name__ == '__main__':
    model = create_model()
    model.summary()