# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 12:33:43 2023

@author: utkarsha
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
import tensorflow as tf
from PIL import Image
from skimage import measure, color, io
import cv2
import matplotlib.pyplot as plt
from skimage import transform
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

IMG_HEIGHT = 128
IMG_WIDTH = 128
MODEL = tf.keras.models.load_model("../models/class/1")
MODEL2 = tf.keras.models.load_model("../models/seg/1")


def read_file_as_image(data):
    image = (Image.open(BytesIO(data)))
    return image


def segment(images):
    segmented = (MODEL2.predict(images) > 0.2).astype(np.uint8)

    plt.imsave('output.jpg', segmented.squeeze(), cmap='gray')

    img = cv2.imread('output.jpg')  # Read as color (3 channels)
    img_grey = img[:, :, 0]

    ret1, thresh = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=10)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret3, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 10
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    img[markers == -1] = [0, 255, 255]

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('Overlay on original image', img)
    cv2.imshow('Colored Grains', img2)
    plt.imsave('color.jpg', img2)
    cv2.waitKey(0)

    props = measure.regionprops_table(markers, intensity_image=img_grey,
                                      properties=['label',
                                                  'area', 'equivalent_diameter',
                                                  'mean_intensity', 'solidity'])

    import pandas as pd
    df = pd.DataFrame(props)
    df = df[df.mean_intensity > 100]

    print(df.head())
    df.to_csv('measurement.csv')

class_label = []
with open('../class_dictionary.json', "r") as f:
    __class_name_to_number = json.load(f)
    class_label = {v: k for k, v in __class_name_to_number.items()}


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    np_image = np.array(image).astype('float32') / 255
    np_image = transform.resize(np_image, (128, 128, 1))
    img_batch = np.expand_dims(np_image,0)

    
    predictions = MODEL.predict(img_batch)

    predicted_class = np.argmax(predictions[0])

    predicted_name = class_label[predicted_class]
    confidence = np.max(predictions[0])
    segment(img_batch)

    print("RESULT", predicted_name)

    return {
        'class': predicted_name,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


