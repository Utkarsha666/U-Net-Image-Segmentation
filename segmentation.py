# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:55:55 2023

@author: utkar
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import glob 

import cv2
from unet import unet

##########################################################################################
##########################################################################################
############# ignores gpu memory error ###################################################
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
#######################################################################################
##########################################################################################
seed = 123
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

images_path = 'COVID-19_Radiography_Dataset/*/images/'
mask_path = 'COVID-19_Radiography_Dataset/*/masks/'

image_datasets = []
mask_datasets = []

#########################################################################################################
###################### Loop through a image and masks folder and save in the image_datasets ########################
###########################################################################################################
##########################################################################################################
for filename in glob.glob('COVID-19_Radiography_Dataset/*/images/*.png', recursive=True):
    image = cv2.imread(filename, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_datasets.append(np.array(image))
        
for filename in glob.glob('COVID-19_Radiography_Dataset/*/masks/*.png', recursive=True):
    image = cv2.imread(filename, 0)
    image = Image.fromarray(image)
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    mask_datasets.append(np.array(image))
#######################################################################################################################
#####################################################################################################################
######################################################################################################################

################ plot the figure #####################################################################################
import random
import numpy as np
image_number = random.randint(0, len(image_datasets))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_datasets[image_number], (128, 128)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(mask_datasets[image_number], (128, 128)), cmap='gray')
plt.show()

########################## Normalize the datasets ############################################
#########
image_datasets = np.array(image_datasets).astype('float32') # Memory error with float64 so converted 
mask_datasets = np.array(mask_datasets).astype('float32')
image_datasets = np.expand_dims(np.array(image_datasets), 3) / 255
mask_datasets = np.expand_dims(np.array(mask_datasets), 3)/ 255



######################## Split the training and testing by 10% ##########################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_datasets, mask_datasets, test_size=0.1, random_state=123)

###################################################################################################################

def get_model():
    return unet(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()
history = model.fit(X_train, y_train, batch_size=16, epochs = 20, validation_split = 0.1)
 
model_version=max([int(i) for i in os.listdir("models/seg") + [0]]) + 1
model.save(f"models/seg/{model_version}")

model.save("seg.h5")

y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5
intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)


plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

