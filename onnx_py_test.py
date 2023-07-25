import onnxruntime as ort
import numpy as np
import numpy as np
import cv2

img=cv2.imread('fr.jpg')
img=cv2.resize(img,(640,640))
img=img.transpose(2,0,1)
img=img.reshape(1,3,640,640)
img=img.astype(np.float32)


modelInputShape = [1, 3, 640, 640]
topk = 1
iouThreshold = 0.45
scoreThreshold = 0.25


config = {
    "top_k": topk,
    "iou_threshold": iouThreshold,
    "score_threshold": scoreThreshold,
}


# Load Model
ort_sess = ort.InferenceSession('yolov8n-pose.onnx')
#load nms model
ort_sess_nms = ort.InferenceSession('modified_nms-yolov8.onnx')
# Run inference

outputs = ort_sess.run(None,{'images': img})
selected_indices = ort_sess_nms.run(None,{ "detection": outputs ,"config": config })


print(outputs[0].shape)
print(selected_indices[0])