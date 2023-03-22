
from typing import Text
import yaml
import pandas as pd

def data_load(config_path: Text) -> None:
    # Config
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
        print(config)

    # config["data"]["train_data"]
    train_df = pd.read_csv(config["data"]["train_data"])
    test_df = pd.read_csv(config["data"]["test_data"])

    print("Data Load complete")


