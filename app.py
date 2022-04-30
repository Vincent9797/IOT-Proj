import argparse
import cv2
import os

import numpy as np
from datetime import datetime
import time

import flask
from flask import request, jsonify, make_response
import tensorflow as tf

from utils import get_data, get_model

app = flask.Flask(__name__)
app.config["DEBUG"] = True
SIZE = 128
USER_INPUT_COUNT = 0
RETRAIN_MODEL_COUNT = 4
RETRAIN_MODEL_X = np.zeros((RETRAIN_MODEL_COUNT, SIZE, SIZE, 3))
RETRAIN_MODEL_Y = np.zeros((RETRAIN_MODEL_COUNT, 1))

@app.route('/', methods=['GET'])
def home():
    return "<h1>This is a web server for classifying gender.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    label, retrain = run_inference(request.json["image_data"])

    # if retrain:
    #     retrain_model()

    response = make_response(
        jsonify({"Label": label}),
        200,
    )
    return response

def run_inference(img):
    global MODEL, USER_INPUT_COUNT, RETRAIN_MODEL_COUNT, RETRAIN_MODEL_X, RETRAIN_MODEL_Y

    img = cv2.resize(np.array(img) / 255, (SIZE, SIZE))
    print(f'Starting prediction @ time = ', datetime.now().strftime("%H:%M:%S"))
    y_pred = MODEL.predict(np.expand_dims(img, 0))
    time.sleep(3) # simulate long process
    print(f'Ending prediction @ time = ', datetime.now().strftime("%H:%M:%S"))
    label = "Male" if y_pred[0][0] >= 0.5 else 'Female'
    retrain = False

    RETRAIN_MODEL_X[USER_INPUT_COUNT] = img
    RETRAIN_MODEL_X[USER_INPUT_COUNT, 0] = 1 if y_pred[0][0] >= 0.5 else 0

    USER_INPUT_COUNT += 1
    if USER_INPUT_COUNT % RETRAIN_MODEL_COUNT == 0:
        retrain = True
        USER_INPUT_COUNT = 0

    return label, retrain

def train_model():
    x_train, x_val, y_train, y_val = get_data(size=SIZE)
    model = get_model(size=SIZE)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=10,
              validation_data=(x_val, y_val),
              batch_size=16)
    model.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model.h5'))
    return model

def retrain_model():
    global MODEL, RETRAIN_MODEL_X, RETRAIN_MODEL_Y

    tf.keras.backend.set_value(MODEL.optimizer.learning_rate, 1e-5)
    MODEL.fit(RETRAIN_MODEL_X, RETRAIN_MODEL_Y,
              epochs=10)

    RETRAIN_MODEL_X[:,:,:,:] = 0
    RETRAIN_MODEL_Y[:,:] = 0
    print('RETRAINED MODEL')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if not args.train:
        MODEL = tf.keras.models.load_model('model.h5')
    else:
        MODEL = train_model()
    app.run(threaded=True)
