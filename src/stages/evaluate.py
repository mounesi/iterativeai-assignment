import yaml
import numpy as np
from keras.models import load_model
from typing import Text

def evaluate(config_path: Text) -> None:
    # Config
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Load the test data from files
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    # Load the model from the file
    model_path = "models/my_model.h5"
    model = load_model(model_path)

    # Evaluate the model
    print("Evaluating the model...")
    print("")
    scores = model.evaluate(x_test, y_test, verbose=0)

    print("Test Loss: {:.4f}".format(scores[0]))
    print("Test Accuracy: {:.4f}".format(scores[1]))

    print("Evaluate completed.")
