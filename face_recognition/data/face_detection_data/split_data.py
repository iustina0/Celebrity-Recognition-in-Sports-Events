

import cv2
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split



import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

csv_filename = 'face_recognition\\data\\face_detection_data\\faces.csv'
df = pd.read_csv(csv_filename)

images_folder = 'face_recognition\\data\\face_detection_data\\images\\'

train_folder = 'face_recognition\\data\\face_detection_data\\train\\images\\'
test_folder = 'face_recognition\\data\\face_detection_data\\test\\images\\'
val_folder = 'face_recognition\\data\\face_detection_data\\val\\images\\'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42)
test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

def move_images_and_create_csv(df, folder):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, 'labels.csv'), index=False)
    for filename in df['image_name']:
        source = os.path.join(images_folder, filename)
        destination = os.path.join(folder, filename)
        shutil.copy(source, destination)

train_grouped = train_df.groupby('image_name')
for _, group_df in train_grouped:
    move_images_and_create_csv(group_df, train_folder)

move_images_and_create_csv(test_df, test_folder)

move_images_and_create_csv(val_df, val_folder)
