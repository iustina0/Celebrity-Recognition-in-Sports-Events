import json
import codecs
import tarfile
import cv2
import numpy as np
import tensorflow as tf
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import albumentations as alb

# Load image
# img = cv2.imread("face_recognition\\data\\face_detection_data\\train\\images\\00002983.jpg")
# print(img.shape)

# image_filename = '00002983.jpg' 

# # Load CSV file
# csv_filename = 'face_recognition\\data\\face_detection_data\\faces.csv'
# df = pd.read_csv(csv_filename)
# row = df[df['image_name'] == image_filename]

# # Get image dimensions
# img_height, img_width, _ = img.shape

# # Define the minimum crop size
# min_crop_height = 450
# min_crop_width = 450

# # Check if the image size is larger than the minimum crop size
# if img_height > min_crop_height and img_width > min_crop_width:
#     augmentator.transforms.insert(0, alb.RandomCrop(height=min_crop_height, width=min_crop_width))

# # Apply augmentations
# augmented = augmentator(image=img, bboxes=[[row["x0"].iloc[0], row["y0"].iloc[0], row["x1"].iloc[0], row["y1"].iloc[0]]],
#                         class_labels=["face"])
# print(augmented)
# if len(augmented["bboxes"]) != 0:

# # Draw bounding box
#     bbox = augmented["bboxes"][0]
#     cv2.rectangle(augmented["image"],
#                 (int(bbox[0]), int(bbox[1])),
#                 (int(bbox[2]), int(bbox[3])),
#                 (255, 0, 0), 2)

# # Display augmented image
# plt.imshow(cv2.cvtColor(augmented["image"], cv2.COLOR_BGR2RGB))
# plt.show()
# # from torchvision.io import read_image, ImageReadMode


# augmentator = alb.Compose([
#     alb.HorizontalFlip(p=0.5),
#     alb.RandomBrightnessContrast(p=0.2),
#     alb.RandomGamma(p=0.2),
#     alb.RGBShift(p=0.2),
#     alb.VerticalFlip(p=0.5)
# ], bbox_params=alb.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

# # Iterate over partitions
# for partition in ['train', 'test', 'val']:
#     csv_filename = os.path.join('face_recognition\\data\\face_detection_data', partition, "labels.csv")
#     df = pd.read_csv(csv_filename)
    
#     annotations_data = []  # List to store annotations for the current partition
    
#     for _, face_object in df.iterrows():
#         image_path = os.path.join('face_recognition\\data\\face_detection_data', partition, 'images', face_object["image_name"])
#         img = cv2.imread(image_path)
        
#         coords = [face_object["x0"], face_object["y0"], face_object["x1"], face_object["y1"]]
        
#         try:
#             img_height, img_width, _ = img.shape
            
#             # Augmentation based on image size
#             if img_height > 450 and img_width > 450:
#                 augmentator.transforms.insert(0, alb.RandomCrop(height=450, width=450))
#             elif img_height > 450 and img_width <= 450:
#                 augmentator.transforms.insert(0, alb.RandomCrop(height=450, width=img_width-1))
#             elif img_width > 450 and img_height <= 450:
#                 augmentator.transforms.insert(0, alb.RandomCrop(height=img_height-1, width=450))

#             augmented = augmentator(image=img, bboxes=[coords], class_labels=["face"])
#             augmented_image_path = os.path.join("face_recognition\\data\\aug_data", partition, "images", face_object["image_name"])
#             cv2.imwrite(augmented_image_path, augmented["image"])

#             annotation = {
#                 'image_name': face_object["image_name"],
#                 'bbox_x0': augmented['bboxes'][0][0] if len(augmented['bboxes']) > 0 else "",
#                 'bbox_y0': augmented['bboxes'][0][1] if len(augmented['bboxes']) > 0 else "",
#                 'bbox_x1': augmented['bboxes'][0][2] if len(augmented['bboxes']) > 0 else "",
#                 'bbox_y1': augmented['bboxes'][0][3] if len(augmented['bboxes']) > 0 else "",
#                 'class_label': 1 if len(augmented['bboxes']) > 0 else 0
#             }
#             annotations_data.append(annotation)

#         except Exception as e:
#             print(f'Exception for {face_object["image_name"]}: {e}')

#     # Convert annotations_data list to DataFrame
#     annotations_df = pd.DataFrame(annotations_data)
    
#     # Save annotations DataFrame to CSV for the current partition
#     annotations_csv_path = os.path.join("face_recognition\\data\\aug_data", partition, "annotations.csv")
#     annotations_df.to_csv(annotations_csv_path, index=False)

# import seaborn as sns
# from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
# from torchmetrics.functional.detection import intersection_over_union
# from torchvision.utils import draw_bounding_boxes

# from torchvision.models.detection.ssd import SSDClassificationHead
# from torchvision.models.detection import _utils
# from torchvision.models.detection import SSD300_VGG16_Weights
# from torchvision.ops import box_iou, complete_box_iou, complete_box_iou_loss



# jsonData = []
# images = tf.data.Dataset.list_files('face_recognition\\data\\face_detection_data\\images\\*.jpg', shuffle = False)

# def load_image(path):
#     byte_img = tf.io.read_file(path)
#     img = tf.io.decode_jpeg(byte_img)
#     return img
# IMG_SIZE = 640
# generator = torch.Generator().manual_seed(202)
# BATCH_SIZE = 8

# transformer = transforms.Compose([transforms.ToTensor(),
#                        transforms.Resize((IMG_SIZE,IMG_SIZE)),
#                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# images = []
# for data in tqdm(jsonData):
#     try:
#         response = requests.get(data['content'])
#         img = np.asarray(Image.open(BytesIO(response.content)))
#         images.append([img, data["annotation"]])
#     except Exception as e:
#         print(f"Error processing image: {e}")







