from AI.create_model import create_model
from AI.train_model import checkpoint_path

from PIL import Image


def load_model():
    model = create_model()
    model.load_weights(checkpoint_path)

    return model


