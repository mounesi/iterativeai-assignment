

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau


# Loading the ASL dataset
train_df = pd.read_csv("./data/sign_mnist_train/sign_mnist_train.csv")
test_df = pd.read_csv("./data/sign_mnist_test/sign_mnist_test.csv")

test = pd.read_csv("./data/sign_mnist_test/sign_mnist_test.csv")
y = test['label']

#print(train_df['label'])

# # Data Visuallization and Processing
# plt.figure(figsize = (10,10)) # Label Count
# sns.set_style("darkgrid")
# sns.countplot(train_df['label'])
# plt.show()
# plt.figure(figsize=(10, 10))
# sns.set_style("darkgrid")
# sns.countplot(train_df['label'], order=train_df['label'].value_counts().index)
# plt.show()

#import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

# Convert 'train_df['label']' to a pandas Series
label_series = pd.Series(train_df['label'])
plt.figure(figsize=(10, 10))
sns.set_style("darkgrid")
# Plot the countplot with the ordered categories
sns.countplot(x=label_series, order=label_series.value_counts().index)
#plt.show()


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


f, ax = plt.subplots(2,5) 
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()

# Display the plots    
#plt.show()

# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)


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

history = model.fit(datagen.flow(x_train,y_train, batch_size = 128) ,epochs = 4 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])


print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

# epochs = [i for i in range(20)]
# fig , ax = plt.subplots(1,2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)

# ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
# ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
# ax[0].set_title('Training & Validation Accuracy')
# ax[0].legend()
# ax[0].set_xlabel("Epochs")
# ax[0].set_ylabel("Accuracy")

# ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
# ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
# ax[1].set_title('Testing Accuracy & Loss')
# ax[1].legend()
# ax[1].set_xlabel("Epochs")
# ax[1].set_ylabel("Loss")
# plt.show()

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

# # Display the plots
# plt.show()

# import matplotlib.pyplot as plt

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)
   
# print(train_acc.shape) # Print the shape of 'train_acc' to debug the error

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

# # Display the plots
# plt.show()

# import matplotlib.pyplot as plt

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)

# print(len(train_acc)) # Print the length of 'train_acc' to debug the error

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

## Display the plots
#plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = np.array(history.history['accuracy']).flatten()
# train_loss = np.array(history.history['loss']).flatten()
# val_acc = np.array(history.history['val_accuracy']).flatten()
# val_loss = np.array(history.history['val_loss']).flatten()
# fig.set_size_inches(16,9)

# print(train_acc.shape) # Print the shape of 'train_acc' to debug the error

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

# # Display the plots
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)

# train_acc = np.squeeze(train_acc)  # remove the single dimension from 'train_acc'
# print(train_acc.shape) # Print the shape of 'train_acc' to debug the error

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

# # Display the plots
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# epochs = [i for i in range(20)]
# fig, ax = plt.subplots(1, 2)
# train_acc = history.history['accuracy']
# train_loss = history.history['loss']
# val_acc = history.history['val_accuracy']
# val_loss = history.history['val_loss']
# fig.set_size_inches(16,9)

# train_acc = np.squeeze(train_acc)  # remove the single dimension from 'train_acc'
# print(train_acc.shape) # Print the shape of 'train_acc' to debug the error

# # Plot training accuracy on the first axis of 'ax'
# ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
# ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('Accuracy/Loss')
# ax[0].legend(loc='best')

# # Plot validation accuracy on the second axis of 'ax'
# ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
# ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Accuracy/Loss')
# ax[1].legend(loc='best')

# # Display the plots
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

epochs = [i for i in range(4)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

train_acc = np.ravel(train_acc)  # flatten the 'train_acc' array
print(train_acc.shape) # Print the shape of 'train_acc' to debug the error

# Plot training accuracy on the first axis of 'ax'
ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, train_loss, 'bo-', label='Training Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy/Loss')
ax[0].legend(loc='best')

# Plot validation accuracy on the second axis of 'ax'
ax[1].plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
ax[1].plot(epochs, val_loss, 'yo-', label='Validation Loss')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy/Loss')
ax[1].legend(loc='best')

# Display the plots
plt.show()


# predictions = model.predict_classes(x_test)
# for i in range(len(predictions)):
#     if(predictions[i] >= 9):
#         predictions[i] += 1
# predictions[:5]   


# # Obtain model predictions
# predictions = model.predict(x_test)

# # Get the class labels with highest probability
# predicted_labels = np.argmax(predictions, axis=1)

# # Adjust predicted labels
# for i in range(len(predicted_labels)):
#     if predicted_labels[i] >= 9:
#         predicted_labels[i] += 1

# predicted_labels[:5]   # print first 5 predicted labels


# classes = ["Class " + str(i) for i in range(25) if i != 9]
# print(classification_report(y, predictions, target_names = classes))

# Obtain model predictions
predictions = model.predict(x_test)

# Get the class labels with highest probability
predicted_labels = np.argmax(predictions, axis=1)

# Adjust predicted labels
for i in range(len(predicted_labels)):
    if predicted_labels[i] >= 9:
        predicted_labels[i] += 1

# Convert predicted labels to integer data type
predicted_labels = predicted_labels.astype(int)

# Print classification report
classes = ["Class " + str(i) for i in range(25) if i != 9]
#print(classification_report(y_test, predicted_labels, target_names=classes))

print(y_test)
print(predicted_labels)
print(classes)




cm = confusion_matrix(y,predictions)


cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])


plt.figure(figsize = (15,15))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')


correct = np.nonzero(predictions == y)[0]


i = 0
for c in correct[:6]:
    plt.subplot(3,2,i+1)
    plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1

plt.show()