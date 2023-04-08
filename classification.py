# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 21:29:31 2023

@author: utkar
"""


import tensorflow as tf

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
from tensorflow.keras import layers, Model, Sequential
##########################################################################################
##########################################################################################
############# ignores gpu memory error ###################################################
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
#######################################################################################
##########################################################################################
########################################################################################################
seed = 123
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

images_path = 'COVID-19_Radiography_Dataset/'
image_datasets = []
class_label = []

print(os.listdir(images_path))

class_label_to_name = {
    'COVID': 0,
    'Lung Opacity': 1,
    'Normal': 2,
    'Viral Pneumonia': 3
    }

############### COVID DATA ################################################
for filename in os.scandir(images_path+"COVID/images/"):
    class_label.append(0)
    image = cv2.imread(filename.path, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_datasets.append(np.array(image))
    
################# LUNG OPACITY DATA #################################
for filename in os.scandir(images_path+"Lung_Opacity/images/"):
    class_label.append(1)
    image = cv2.imread(filename.path, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_datasets.append(np.array(image))
    
############### NORMAL DATA ##########################################
for filename in os.scandir(images_path+"Normal/images/"):
    class_label.append(2)
    image = cv2.imread(filename.path, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_datasets.append(np.array(image))

############ Viral Pneumonia ############################################
for filename in os.scandir(images_path+"Viral Pneumonia/images/"):
    class_label.append(3)
    image = cv2.imread(filename.path, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_datasets.append(np.array(image))
##########################################################################   
########################################################################
## Plot a graph randomly ############################
import random
import numpy as np
image_number = random.randint(0, len(image_datasets))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title(class_label[image_number])
plt.imshow(np.reshape(image_datasets[image_number], (128, 128)), cmap='gray')
plt.show()

############################# Normalization ##################################################
############################################################################################

image_datasets = np.expand_dims(np.array(image_datasets), 3) / 255
class_label = np.array(class_label)

##################################################################################################
####################################################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_datasets, class_label, test_size=0.2, random_state=100, stratify=class_label)

###################################################################################################
####################################################################################################
input_shape = (16, IMG_HEIGHT, IMG_WIDTH, 1) 
model = Sequential([
    layers.Conv2D(32, kernel_size= (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model.build(input_shape=input_shape)
model.summary()
##########################################################################################
model.compile(
    optimizer='adam',
    loss = tf.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    epochs=20,
)

##################################################################################################
scores = model.evaluate(X_test, y_test)
print(scores)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(range(20), acc, label='Training Accuracy')
plt.plot(range(20), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Trainign Vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(20), loss, label='Training Loss')
plt.plot(range(20), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

################################################################################################
y_pred = model.predict(X_test)
print(class_label[np.argmax(y_pred[6])])
print(y_test)
#######################################################################################
import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_label_to_name))
    
##########################################################################################
import os
model_version=max([int(i) for i in os.listdir("models/class") + [0]])+1
model.save(f"models/class/{model_version}")

model.save('class.h5')
