import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import pandas as pd

# Define your CNN architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 56 * 56, 256) 
        self.fc2 = torch.nn.Linear(256, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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



train_df, valid_df, test_df = train_valid_test_split(faces_csv='face_recognition\\data\\face_detection_data\\faces.csv')

# Prepare your dataset and dataloaders
train_dataset = Faces(train_df, width, height)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate your CNN model
model = SimpleCNN()

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, targets in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
