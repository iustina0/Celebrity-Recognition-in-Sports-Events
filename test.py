import json
import os
import glob
import shutil
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as alb
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16

def train_valid_test_split(faces_csv=None, split=0.15):
    all_df = pd.read_csv(faces_csv)
    all_df = all_df.sample(n=500, random_state=7).sample(frac=1)
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

def build_model(input_shape=(120, 120, 3)):
    input_layer = Input(shape=input_shape)
    
    # Use VGG16 as the base convolutional backbone
    vgg = VGG16(include_top=False, input_tensor=input_layer)
    
    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg.output)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid', name='class_output')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg.output)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid', name='regression_output')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

facetracker = build_model()
facetracker.summary()

# Define optimizer and learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size

# Define loss functions
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

train_df, valid_df, test_df = train_valid_test_split(faces_csv='face_recognition\\data\\face_detection_data\\faces.csv')

train_dataset = Faces(train_df, 450, 450)
valid_dataset = Faces(valid_df, 450, 450)
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=10)

hist = model.fit(train_loader, epochs=10, validation_data=valid_loader, callbacks=[tensorboard_callback])
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()