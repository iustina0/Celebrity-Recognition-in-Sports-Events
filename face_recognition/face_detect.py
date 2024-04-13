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
from numba import jit, cuda 

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


class Finding_losses:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class FaceDetectionNN:
    def __init__(self):
        self.BATCH_SIZE = 4
        self.RESIZE_TO = 450
        self.CLASSES = ['background', 'Face']
        self.NUM_CLASSES = 2

        train_df, valid_df, test_df = train_valid_test_split(faces_csv='face_recognition\\data\\face_detection_data\\faces.csv')

        self.train_dataset = Faces(train_df, self.RESIZE_TO, self.RESIZE_TO)
        self.valid_dataset = Faces(valid_df, self.RESIZE_TO, self.RESIZE_TO)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.NUM_CLASSES)
        self.model = self.model.to(DEVICE)

        self.NUM_EPOCHS = 5
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.train_loss_hist = Finding_losses()
        self.val_loss_hist = Finding_losses()
        self.train_itr = 1
        self.val_itr = 1
        self.train_loss_list = []
        self.val_loss_list = []
        self.MODEL_NAME = 'Human Faces Detection'

    def visualize_sample(self, image, target):
        for box in target['boxes']:
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 1
            )

        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def validate(self):
        print('Validating')

        prog_bar = tqdm(self.valid_loader, total=len(self.valid_loader))

        for i, data in enumerate(prog_bar):
            images, targets = data
            images = list(torch.from_numpy(image).to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.val_loss_list.append(loss_value)
            self.val_loss_hist.send(loss_value)
            self.val_itr += 1
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return self.val_loss_list

    def train(self):
        print('Training')

        prog_bar = tqdm(self.train_loader, total=len(self.train_loader))

        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()
            images, targets = data
            images = list(torch.from_numpy(image).to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.train_loss_list.append(loss_value)
            self.train_loss_hist.send(loss_value)
            losses.backward()
            self.optimizer.step()
            self.train_itr += 1
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return self.train_loss_list


def collate_fn(batch):
    return tuple(zip(*batch))


# network = FaceDetectionNN()

# for epoch in range(network.NUM_EPOCHS):
#     print(f"\nEPOCH {epoch+1} of {network.NUM_EPOCHS}")
#     network.train_loss_hist.reset()
#     network.val_loss_hist.reset()
#     start = time.time()
#     train_loss = network.train()
#     val_loss = network.validate()
#     print(f"Epoch #{epoch+1} train loss: {network.train_loss_hist.value:.3f}")
#     print(f"Epoch #{epoch+1} validation loss: {network.val_loss_hist.value:.3f}")
#     end = time.time()
#     print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")

#     if (epoch+1) == network.NUM_EPOCHS:
#         figure_1, train_ax = plt.subplots()
#         figure_2, valid_ax = plt.subplots()
#         train_ax.plot(train_loss, color='blue')
#         train_ax.set_xlabel('iterations')
#         train_ax.set_ylabel('train loss')
#         valid_ax.plot(val_loss, color='red')
#         valid_ax.set_xlabel('iterations')
#         valid_ax.set_ylabel('validation loss')
#         figure_1.savefig(f"train_loss_{epoch+1}.png")
#         figure_2.savefig(f"valid_loss_{epoch+1}.png")
#         torch.save(network.model.state_dict(), f"model{epoch+1}.pth")

# train_ax.plot(train_loss, color='blue')
# train_ax.set_xlabel('iterations')
# train_ax.set_ylabel('train loss')

# valid_ax.plot(val_loss, color='red')
# valid_ax.set_xlabel('iterations')
# valid_ax.set_ylabel('validation loss')
# plt.show()


def draw_red_triangle(img, box, resize_to = [450, 450], frame_shape = [1280, 720]):
    ratio_y = frame_shape[1] / resize_to[1]
    ratio_x = frame_shape[0] / resize_to[0]
    box_scaled = [int(coord * ratio_x) if idx % 2 == 0 else int(coord * ratio_y) for idx, coord in enumerate(box)]
    pts = np.array([[box_scaled[0], box_scaled[1]], [box_scaled[0], box_scaled[3]], [box_scaled[2], box_scaled[3]], [box_scaled[2], box_scaled[1]]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

num_classes = 2 
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


model_path = "model5.pth" 
model.load_state_dict(torch.load(model_path))

model.eval()

def process_image(image_path):
    image = cv2.imread(image_path)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image_resized = cv2.resize(image_RGB, (450, 450))
    image_resized /= 255.0
    image_resized = np.transpose(image_resized, (2, 0, 1))

    image_tensor = torch.from_numpy(image_resized).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    print(prediction)

    threshold = 0.5 

    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    valid_boxes = pred_boxes[pred_scores > threshold]

    for b in valid_boxes:
        box = b.astype(int)
        draw_red_triangle(image, box, [450, 450], [image.shape[1], image.shape[0]])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# process_image("face_recognition\\data\\face_detection_data\\images\\00003431.jpg")

 
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('processed_video.avi', fourcc, 25.0, (1280, 720))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame_RBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            frame_resized = cv2.resize(frame_RBG, (450, 450))
            frame_resized /= 255.0
            frame_resized = np.transpose(frame_resized, (2, 0, 1))

            input_tensor = torch.from_numpy(frame_resized).unsqueeze(0)

            model.eval()
            model.zero_grad()

            with torch.no_grad():
                prediction = model(input_tensor)

            threshold = 0.5 
            pred_boxes = prediction[0]['boxes'].cpu().numpy()
            pred_scores = prediction[0]['scores'].cpu().numpy()

            valid_boxes = pred_boxes[pred_scores > threshold]
            for box in valid_boxes:
                box = box.astype(int)
                draw_red_triangle(frame, box)

            out.write(frame)
        else:
            break

    cap.release()
    out.release()



process_video('video1.avi')