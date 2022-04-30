import requests
import cv2
import matplotlib.pyplot as plt

import numpy as np
if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # camera captures in BGR
    plt.imshow(image)
    plt.show()
    # image = np.zeros((240, 240, 3))
    COUNT = 0
    try:
        response = requests.post('http://localhost:5000/predict', json={'image_data': image.tolist()})
        print(response.json())
    except Exception as e:
        print(e)


