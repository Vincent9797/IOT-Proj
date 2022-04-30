import requests
import cv2
import matplotlib.pyplot as plt
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cap', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)

    if args.cap:
        return_value, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # camera captures in BGR
    else:
        image = np.ones((240, 240, 3))
    plt.imshow(image)
    plt.show()
    COUNT = 0

    try:
        response = requests.post('http://localhost:5000/predict', json={'image_data': image.tolist()})
        print(response.json())
    except Exception as e:
        print(e)


