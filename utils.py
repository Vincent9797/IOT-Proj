import random
from glob import glob
import os

from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf

def get_data(size):
    x_train, y_train = [], []
    x_val, y_val = [], []
    for i in tqdm(glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/male/*.jpg'))):
        img = cv2.imread(i)
        img = cv2.resize(img, (size, size)) / 255
        if random.random() > 0.8:
            x_val.append(img)
            y_val.append([1])
        else:
            x_train.append(img)
            y_train.append([1])

    for i in tqdm(glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/female/*.jpg'))):
        img = cv2.imread(i)
        img = cv2.resize(img, (size, size)) / 255
        if random.random() > 0.8:
            x_val.append(img)
            y_val.append([0])
        else:
            x_train.append(img)
            y_train.append([0])

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    np.random.shuffle(x_train)
    np.random.shuffle(y_train)
    np.random.shuffle(x_val)
    np.random.shuffle(y_val)

    return x_train, x_val, y_train, y_val

def get_model(size):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())
    return model