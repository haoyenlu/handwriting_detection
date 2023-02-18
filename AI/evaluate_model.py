import matplotlib.pyplot as plt

from AI.create_model import create_model
from AI.get_dataset import get_dataset
from AI.train_model import checkpoint_path

if __name__ == "__main__":
    model = create_model()
    (x_train,y_train),(x_test,y_test) = get_dataset()

    # untrained model
    loss , acc = model.evaluate(x_test,y_test,verbose = 2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


    # trained model
    model.load_weights(checkpoint_path)
    loss , acc = model.evaluate(x_test,y_test,verbose = 2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))