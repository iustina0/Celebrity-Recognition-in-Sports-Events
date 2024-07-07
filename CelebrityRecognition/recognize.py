from FaceDetection import face_detect
from FaceRecognition import face_recogniton
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class Face:
    def __init__(self, image, bbox, identity=None):
        self.image = image
        self.bbox = bbox
        self.identity = identity

def crop_to_aspect_ratio(image, bbox, target_ratio=0.725):
    image_height, image_width = image.shape[:2]
    
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    width = x4 - x1
    height = y3 - y4

    new_width = int(height * target_ratio)

    if new_width < width:
        delta = (width - new_width) // 2
        buffer_height = height // 20
        buffer_width = new_width // 20
        
        y1 = max(0, y1 - buffer_height)
        y2 = min(image_height, y2 + buffer_height)
        y3 = min(image_height, y3 + buffer_height)
        y4 = max(0, y4 - buffer_height)

        x1 = max(0, x1 + delta - buffer_width)
        x2 = max(0, x2 + delta - buffer_width)
        x3 = min(image_width, x3 - delta + buffer_width)
        x4 = min(image_width, x4 - delta + buffer_width)

    min_x = max(0, int(min(x1, x2, x3, x4)))
    max_x = min(image_width, int(max(x1, x2, x3, x4)))
    min_y = max(0, int(min(y1, y2, y3, y4)))
    max_y = min(image_height, int(max(y1, y2, y3, y4)))

    cropped_image = image[min_y:max_y, min_x:max_x]
    cropped_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    return cropped_image, cropped_box

image_path = "misc/old/chris.jpg"
image = cv2.imread(image_path)
boxes = face_detect.detect_face(image)
boxes = np.array(boxes, dtype=int)

faces = []
cropped_boxes = []

for box in boxes:
    cropped_face, cropped_box = crop_to_aspect_ratio(image, box)
    faces.append(cropped_face)
    cropped_boxes.append(cropped_box)

for face, box in zip(faces, cropped_boxes):
    identity, distance = face_recogniton.recognize_faces([Image.fromarray(face).convert('RGB')])
    print(f"Identity: {identity}, Distance: {distance}")
    cv2.line(image, box[0], box[1], (0, 0, 255), 2)
    cv2.line(image, box[1], box[2], (0, 0, 255), 2)
    cv2.line(image, box[2], box[3], (0, 0, 255), 2)
    cv2.line(image, box[3], box[0], (0, 0, 255), 2)

    cv2.putText(image, identity[0], (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red color text

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()