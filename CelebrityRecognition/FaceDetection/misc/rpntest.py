import torch
import numpy as np
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from model.RPN import RegionProposalNetwork
from torch import nn

from torchvision.models import vgg16
from torchvision import transforms
from model.fasterRCNN import FasterRCNN
from model.trainer import FasterRCNNTrainer

def load_image_and_features(image_path, extractor):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0) 

    image_tensor = image_tensor.to('cuda')
    features = extractor(image_tensor)
    print(features.shape)
    image_size = image.shape[:2]
    return features, image_size, image

def visualize_proposals(image, rois, scores):
    num_rois = int(rois.shape[0] * 0.05)
    selected_rois = np.random.choice(rois.shape[0], num_rois, replace=False)
    for roi_idx in selected_rois:
        roi = rois[roi_idx]
        x1, y1, x2, y2 = roi.astype(int)
        cv2.rectangle(image, (y1, x1), (y2, x2), (0, 250, 0), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

model = FasterRCNN()
trainer = FasterRCNNTrainer(model).cuda()
trainer.load(".\\FaceDetection\\checkpoints\\fasterrcnn_pretrained-06121613.pth")
faster_rcnn = trainer.faster_rcnn
rpn = faster_rcnn.rpn
extractor = faster_rcnn.extractor

image_path = 'TH110-713_2019_130028 - Copy - Copy.png' 
def draw_red_triangle(img, box):
    box = box.astype(int)
    pts = np.array([[box[1], box[0]], [box[1], box[2]], [box[3], box[2]], [box[3], box[0]]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)



image_path = 'TH110-713_2019_130028 - Copy - Copy.png'
image1 = cv2.imread(image_path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image = np.asarray(image1, dtype=np.float32)
image = image.transpose((2, 0, 1))

boxes, labels, scores = faster_rcnn.predict([image], visualize=True)
boxes = boxes[0]
print(boxes)
for box in boxes:
    draw_red_triangle(image1, box)

plt.imshow(image1)
plt.axis('off')
plt.show()
