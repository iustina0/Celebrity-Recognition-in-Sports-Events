import random
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import os
import numpy as np
from utils.config import opt
from torch.utils.data import Dataset

class EvalDataset(Dataset):
    def __init__(self, pairs_path='FaceRecognition/data/csvs/pairs.csv', transform=None):

        super(EvalDataset, self).__init__()
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        self.transform = transform
        self.identity_to_image_ids = self._parse_identity_file()
        self.pairs_path = pairs_path
        self.validation_images = self.get_paths()

    def _parse_identity_file(self):
        identity_to_image_ids = {}
        with open(opt.identity_file, 'r') as f:
            for line in f:
                image_id, identity = line.strip().split()
                if identity not in identity_to_image_ids:
                    identity_to_image_ids[identity] = []
                identity_to_image_ids[identity].append(image_id)
        return identity_to_image_ids
    
    def read_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split(',')
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_identity_from_image_id(self, image_id):
        for identity, image_ids in self.identity_to_image_ids.items():
            if image_id in image_ids:
                return identity
        return None

    def get_paths(self):
        pairs = self.read_pairs(self.pairs_path)
        pairs = list(pairs)
        pairs = random.sample(pairs, 1000)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if random.random() < 0.5:
                path0 = os.path.join(opt.image_dir, self.get_identity_from_image_id(pair[0]), pair[0])
                path1 = os.path.join(opt.image_dir, self.get_identity_from_image_id(pair[1]), pair[1])
                issame = True
            else: 
                path0 = os.path.join(opt.image_dir, self.get_identity_from_image_id(pair[0]), pair[0])
                path1 = os.path.join(opt.image_dir, self.get_identity_from_image_id(pair[2]), pair[2])
                issame = False

            if os.path.exists(path0) and os.path.exists(path1): 
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                print(f'Not found: {pair[0]} or {pair[1]}/{pair[2]}')
                nrof_skipped_pairs += 1

        if nrof_skipped_pairs > 0:
            print(f'Skipped {nrof_skipped_pairs} image pairs')

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = self.transform(Image.open(path_1)), self.transform(Image.open(path_2))
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)
    
    
    def display_samples(self, num_samples=5):
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
        for i in range(num_samples):
            img1, img2, issame = self.__getitem__(i)
            img1 = transforms.ToPILImage()(img1)
            img2 = transforms.ToPILImage()(img2)
            
            axes[i, 0].imshow(img1)
            axes[i, 0].axis('off')
            axes[i, 0].set_title(f'Image 1 - {"Same" if issame else "Different"}')

            axes[i, 1].imshow(img2)
            axes[i, 1].axis('off')
            axes[i, 1].set_title(f'Image 2 - {"Same" if issame else "Different"}')
        plt.show()
