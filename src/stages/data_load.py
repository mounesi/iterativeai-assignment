import yaml
import numpy as np
import pandas as pd
import os
from typing import Text

def data_load(config_path: Text) -> None:
    # Config
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    train_df = pd.read_csv(config["data"]["train_data"])
    test_df = pd.read_csv(config["data"]["test_data"])

    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    # Normalize the data
    x_train = x_train / 255
    x_test = x_test / 255

    # Reshaping the data from 1-D to 3-D as required through input by CNN's
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # Create the data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the data to files
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    print("Data load complete.")

