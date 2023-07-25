import time
import onnxruntime as ort
import numpy as np
import cv2

img = cv2.imread('fr.jpg')
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)
img = img.reshape(1, 3, 640, 640)
img = img.astype(np.float32)
img = img / 255.0


ort_sess = ort.InferenceSession('public/model/yolov8n-pose.onnx', providers=['CPUExecutionProvider'])


while True:
    start = time.time()
    outputs = ort_sess.run(None, {'images': img})
    end = time.time()
    print('time: ', end - start)
