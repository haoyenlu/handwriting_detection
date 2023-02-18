from AI.create_model import create_model



def load_model():
    model = create_model()
    model.load_weights("AI/training_1/cp.pkt")

    return model

