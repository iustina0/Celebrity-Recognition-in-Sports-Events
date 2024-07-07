import cv2
import numpy as np
import pandas as pd
import os
import glob
import torch as t
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from FaceDetection.utils.config import opt


class FaceDetectionDataset:
    def __init__(self, split='trainval'):
        self.data_dir = os.path.join(opt.data_dir, "images")

        faces_csv = os.path.join(opt.data_dir, "faces.csv")

        all_df = pd.read_csv(faces_csv)
        all_df = all_df[(all_df['width'] <= 1200) & (all_df['height'] <= 1200)]

        all_df = all_df.sample(n=opt.train_valid_num, random_state=7)
        all_df.sample(frac=1)

        len_df = len(all_df)
        train_test_split = int((1 - opt.valid_split) * len_df)
        train_val_df = all_df[:train_test_split]
        test_df = all_df[train_test_split:]
        len_tv_df = len(train_val_df)
        train_val_split = int((1 - opt.valid_split) * len_tv_df)
        train_df = train_val_df[:train_val_split]
        valid_df = train_val_df[train_val_split:]

        if split == 'train':
            self.df = train_df
        elif split == 'val':
            self.df = valid_df
        elif split == 'test':
            self.df = test_df
        elif split == 'trainval':
            self.df = train_val_df

        self.set_image_names = self.df['image_name'].tolist()
        self.image_paths = glob.glob(f"{self.data_dir}/*.jpg")
        self.all_images = [image_path.split('\\')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        self.images = [i for i in self.set_image_names if i in self.all_images]


    def __len__(self):
        return len(self.images)

    def get_example(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.asarray(image, dtype=np.float32)

        boxes = []
        labels = []

        filtered_df = self.df.loc[self.df['image_name'] == image_name]

        for i in range(len(filtered_df)):
            xmin = int(filtered_df['x0'].iloc[i])
            xmax = int(filtered_df['x1'].iloc[i])
            ymin = int(filtered_df['y0'].iloc[i])
            ymax = int(filtered_df['y1'].iloc[i])

            boxes.append([ymin, xmin, ymax, xmax])
            labels.append(0)

        boxes = np.stack(boxes).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)

        return image, boxes, labels

    __getitem__ = get_example


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def preprocess(img, min_size=600, max_size=1000):
    c, h, w = img.shape
    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (c, h * scale, w * scale), mode='reflect', anti_aliasing=False)
    return pytorch_normalize(img)


class Transform(object):
    def __init__(self, min_size=300, max_size=600):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, orig_h, orig_w = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, h, w = img.shape
        scale = h / orig_h
        bbox = resize_bbox(bbox, (orig_h, orig_w), (h, w))

        return img, bbox, label, scale


class TrainDataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = FaceDetectionDataset(split='trainval')
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db[idx]

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test'):
        self.opt = opt
        self.db = FaceDetectionDataset(split=split)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db[idx]
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label

    def __len__(self):
        return len(self.db)