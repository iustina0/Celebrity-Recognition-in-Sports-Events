import numpy as np
import cv2
from FaceDetection.main import FasterRCNNModel

def detect_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))
    
    boxes, _, _ = FasterRCNNModel.predict([image], visualize=True)
    boxes = boxes[0]
    return [[[box[1], box[0]], [box[1], box[2]], [box[3], box[2]], [box[3], box[0]]] for box in boxes]
    

