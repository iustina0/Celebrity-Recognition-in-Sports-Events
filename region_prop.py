import json
import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import cv2
from torch.nn import functional as F
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_valid_test_split(faces_csv=None, split=0.15):
    all_df = pd.read_csv(faces_csv)
    all_df = all_df.sample(n=500, random_state=7)
    all_df.sample(frac=1)
    len_df = len(all_df)
    trainTest_split = int((1-split)*len_df)
    trainVal_df = all_df[:trainTest_split]
    test_df = all_df[trainTest_split:]
    lenTV_df = len(trainVal_df)
    trainVal_split = int((1-split)*lenTV_df)
    train_df = trainVal_df[:trainVal_split]
    valid_df = trainVal_df[trainVal_split:]
    return train_df, valid_df, test_df

class Faces(Dataset):
    def __init__(self, dataset, width, height, dir_path="face_recognition\\data\\face_detection_data\\images\\"):
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.dataset = dataset
        self.set_image_names = self.dataset['image_name'].tolist()
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        self.images = [i for i in self.set_image_names if i in self.all_images]

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        image_resized = np.transpose(image_resized, (2, 0, 1))

        boxes = []
        labels = []

        filtered_df = self.dataset.loc[self.dataset['image_name'] == image_name]

        for i in range(len(filtered_df)):
            xmin = int(filtered_df['x0'].iloc[i])
            xmax = int(filtered_df['x1'].iloc[i])
            ymin = int(filtered_df['y0'].iloc[i])
            ymax = int(filtered_df['y1'].iloc[i])

            image_width = int(filtered_df['width'].iloc[i])
            image_height = int(filtered_df['height'].iloc[i])

            xmin_final = (xmin / image_width) * self.width
            xmax_final = (xmax / image_width) * self.width
            ymin_final = (ymin / image_height) * self.height
            ymax_final = (ymax / image_height) * self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.ones((boxes.shape[0],), dtype=torch.int64) if boxes.shape[0] > 1 else torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx])
        }

        return image_resized, target

    def __len__(self):
        return len(self.set_image_names)

train_df, valid_df, test_df = train_valid_test_split(faces_csv='face_recognition\\data\\face_detection_data\\faces.csv')
train_dataset = Faces(train_df, 450, 450)
valid_dataset = Faces(valid_df, 450, 450)


def create_model(input_shape):
    inputs = Input(input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(5, activation='sigmoid')(x)  # 4 for bounding box, 1 for class label
    model = Model(inputs=inputs, outputs=outputs)
    return model

model = create_model((450, 450, 3))
model.compile(optimizer='adam', loss='mse')

# Define a DataLoader for the training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Convert the datasets to tensorflow format
train_data = tf.data.Dataset.from_generator(lambda: train_loader, output_signature=(
    tf.TensorSpec(shape=(None, 450, 450, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))
valid_data = tf.data.Dataset.from_generator(lambda: valid_loader, output_signature=(
    tf.TensorSpec(shape=(None, 450, 450, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))

# Train the model
history = model.fit(train_data, validation_data=valid_data, epochs=10)