import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from typing import Text
import yaml
import pandas as pd
import numpy as np

def train(config_path: Text) -> None:
    # Config
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Load the training data from files
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")

    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range = 0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False)

    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor = config["train_params"]["RLP"]["monitor"], patience = config["train_params"]["RLP"]["patience"], verbose=config["train_params"]["RLP"]["verbose"] ,factor=config["train_params"]["RLP"]["factor"], min_lr=config["train_params"]["RLP"]["min_lr"])

    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
    model.summary()

    history = model.fit(datagen.flow(x_train,y_train, batch_size = config["train"]["fit"]["batch_size"]) ,epochs = config["train"]["fit"]["epochs"] , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])

    model_path = "models/my_model.h5"
    import os
    # Check if the file already exists  
    if os.path.exists(model_path):
        # If it exists, remove it
        os.remove(model_path)
    # Save the model to a file
    model.save(model_path)

    print("train data completed")




