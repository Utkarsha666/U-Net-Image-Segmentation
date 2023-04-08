# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 00:52:36 2023

@author: utkar
"""
############################################################################################################################################
import tensorflow as tf
from unet import unet
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure, color, io
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.models import load_model

IMG_HEIGHT = 128
IMG_WIDTH = 128


def get_model():
    return unet(IMG_HEIGHT, IMG_WIDTH, 1)

model = get_model()

model.load_weights("seg.h5")

def segment(images):

    test_img = cv2.imread(images,0)
    test_img = Image.fromarray(test_img)
    test_img = test_img.resize((IMG_HEIGHT, IMG_WIDTH))
    test_img = np.array(test_img).astype('float32') 
    test_img = np.expand_dims(np.array(test_img), 2) / 255
    test_img = np.expand_dims(np.array(test_img), 0)

    segmented = (model.predict(test_img) > 0.2).astype(np.uint8)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title('Testing Image')
    plt.imshow(test_img.squeeze(), cmap='gray')
    plt.subplot(222)
    plt.title('Segmented Image')
    plt.imshow(segmented.squeeze(), cmap='gray')
    plt.show()
    
    plt.imsave('output.jpg', segmented.squeeze(), cmap='gray')
    
    img = cv2.imread('output.jpg')  #Read as color (3 channels)
    img_grey = img[:,:,0]
    
    ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    sure_bg = cv2.dilate(opening,kernel,iterations=10)
    
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
    
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret3, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers+10
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img, markers)
    
    img[markers == -1] = [0,255,255]  
    
    img2 = color.label2rgb(markers, bg_label=0)
    
    cv2.imshow('Overlay on original image', img)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)
    
    props = measure.regionprops_table(markers, intensity_image=img_grey, 
                                  properties=['label',
                                              'area', 'equivalent_diameter',
                                              'mean_intensity', 'axis_major_length', 'axis_minor_length'])
    
    import pandas as pd
    df = pd.DataFrame(props)
    df = df[df.mean_intensity > 100]
    
    print(df.head())
    
segment('F1.large.jpg')

def cnn(IMG_HEIGHT, IMG_WIDTH, num_channels=1):
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
    model.compile(
        optimizer='adam',
        loss = tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return cnn

def get_class_model():
    return cnn(IMG_HEIGHT, IMG_WIDTH, 1)

#class_model = tf.keras.models.load_model("models/class/1")
class_model = load_model("class.h5", compile="False")
class_label = []
import json

with open('class_dictionary.json', "r") as f:
    __class_name_to_number = json.load(f)
    class_label = {v:k for k, v in __class_name_to_number.items()}

def predict(images):
    test_img = cv2.imread(images,0)
    test_img = Image.fromarray(test_img)
    test_img = test_img.resize((IMG_HEIGHT, IMG_WIDTH))
    test_img = np.array(test_img).astype('float32') 
    test_img = np.expand_dims(np.array(test_img), 2) / 255
    test_img = np.expand_dims(np.array(test_img), 0)
    
    y_pred = (class_model.predict(test_img))
    print(class_label[np.argmax(y_pred[0])])
    
    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.title(class_label[np.argmax(y_pred[0])])
    plt.imshow(test_img.squeeze(), cmap='gray')
    plt.show()
    
predict('F1.large.jpg')
    
