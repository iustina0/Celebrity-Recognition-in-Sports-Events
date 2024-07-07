import csv
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from utils.config import opt
from torch.nn.modules.distance import PairwiseDistance
from model.NN2Inception import NN2Inception #, NN2Inception
from FaceRecognition.model.incept import Resnet34Triplet 
def load_model(epoch):
    path=f'model_training_checkpoints/model_triplet_epoch_{epoch-1}.pt'
    checkpoint = torch.load(path)
    model = Resnet34Triplet().cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()

class TripletFaceDataset(Dataset):
    def __init__(self,training_dataset_csv_path=opt.parsed_identity_file, image_dir=opt.image_dir, identity_file=opt.identity_file, target_size=(224, 224), triplet_file=opt.triplet_file, batch_size=32, 
                 shuffle=True, num_triplets=2560, num_human_identities_per_batch=32, triplet_batch_size=32, transform=None, epoch=0):
        
        self.df = pd.read_csv(training_dataset_csv_path, dtype={'id': object, 'name': object, 'class': int})
        self.image_dir = image_dir
        self.identity_file = identity_file
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_triplets = num_triplets
        self.num_human_identities_per_batch = num_human_identities_per_batch
        self.triplet_batch_size = triplet_batch_size
        self.transform = transform
        self.epoch = epoch
        self.triplet_file = triplet_file
        self.model = None

        self.id_to_name = dict(zip(self.df['id'], self.df['name']))
        self.id_to_class = dict(zip(self.df['id'], self.df['class']))
        self.name_to_id = dict(zip(self.df['name'], self.df['id']))
        self.class_to_id = dict(zip(self.df['class'], self.df['id']))
        self.class_names = self.df['name'].unique().tolist()

        self.identity_to_image_ids = self._parse_identity_file()
        self.image_ids = [img_id for ids in self.identity_to_image_ids.values() for img_id in ids]
        self.indexes = np.arange(len(self.image_ids))
        self.training_triplets = self.generate_triplets()

    def _parse_identity_file(self):
        identity_to_image_ids = {}
        with open(self.identity_file, 'r') as f:
            for line in f:
                image_id, identity = line.strip().split()
                if identity not in identity_to_image_ids:
                    identity_to_image_ids[identity] = []
                identity_to_image_ids[identity].append(image_id)
        return identity_to_image_ids

    def _find_image_path(self, image_id):
        root = self.get_identity_from_image_id(image_id)
        if root is not None:
            return os.path.join(opt.image_dir,root, image_id)
        return None

    def get_identity_from_image_id(self, image_id):
        for identity, image_ids in self.identity_to_image_ids.items():
            if image_id in image_ids:
                return identity
        return None
    
    def make_dictionary_for_face_class(self):   
        face_classes = dict()
        for idx, label in enumerate(self.df['class']):
            if label not in face_classes:
                face_classes[label] = []
            face_classes[label].append(self.df_dict_id[idx])

        return face_classes
    
    def _get_embedding(self, image_path):
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            embedding = self.model(image).cpu().numpy().flatten()
        return embedding


    
    def gen_triplet_csv(self, num_hard_pairs=0, num_random_pairs=2560, num_semi_hard_pairs=0):
        self.model = load_model(self.epoch)

        l2_distance = PairwiseDistance().to('cuda')

        ids = pd.read_csv('ids.csv')
        valid_names = set(ids['Name'].values)
        data = pd.read_csv(self.identity_file, sep='\s+', names=['image', 'label'])
        grouped = data.groupby('label')['image'].apply(list).to_dict()
        
        with open(self.triplet_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['anchor', 'positive', 'negative'])

            pairs_generated = 0
            with tqdm(total=num_hard_pairs, desc="Generating Hard Pairs", unit="pair") as pbar:
                while pairs_generated < num_hard_pairs:
                    for label, images in grouped.items():
                        if len(images) < 2:
                            continue

                        anchor_img = random.choice(images)
                        anchor_img_path = os.path.join(self.image_dir, label, anchor_img)
                        anchor_embedding = self._get_embedding(anchor_img_path)
                        anchor_embedding = torch.tensor(anchor_embedding, device='cuda')

                        pos_class_embeddings = np.array([self._get_embedding(os.path.join(self.image_dir, label, img)) for img in images]).astype('float32')
                        pos_class_embeddings = torch.tensor(pos_class_embeddings, device='cuda')
                        pos_distances = l2_distance.forward(pos_class_embeddings, anchor_embedding)
                        pos_distances = pos_distances.cpu().numpy() 
                        hard_pos_idx = np.argmax(pos_distances)

                        random_neg_labels = random.sample([key for key in grouped.keys() if key != label], 15)
                        neg_embeddings = []
                        neg_img_paths = []

                        for neg_label in random_neg_labels:
                            neg_img = random.choice(grouped[neg_label])
                            neg_img_path = os.path.join(self.image_dir, neg_label, neg_img)
                            neg_embedding = self._get_embedding(neg_img_path)
                            neg_embeddings.append(neg_embedding)
                            neg_img_paths.append(neg_img)

                        neg_embeddings = np.array(neg_embeddings)
                        neg_embeddings = torch.tensor(neg_embeddings, device='cuda')
                        neg_distances = l2_distance.forward(neg_embeddings, anchor_embedding)
                        neg_distances = neg_distances.cpu().numpy() 
                        hard_neg_idx = np.argmin(neg_distances)

                        hard_neg_image = neg_img_paths[hard_neg_idx]
                        writer.writerow([anchor_img, images[hard_pos_idx], hard_neg_image])

                        pairs_generated += 1
                        pbar.update(1)

                        if pairs_generated >= num_hard_pairs:
                            break


            pairs_generated = 0
            acc_hard = 0
            with tqdm(total=num_semi_hard_pairs, desc="Generating Semi-Hard Pairs", unit="pair") as pbar:
                while pairs_generated < num_semi_hard_pairs:
                    for label, images in grouped.items():
                        if len(images) < 2:
                            continue

                        anchor_img, pos_img = random.sample(images, 2)
                        anchor_img_path = os.path.join(self.image_dir, label, anchor_img)
                        anchor_embedding = self._get_embedding(anchor_img_path)
                        anchor_embedding = torch.tensor(anchor_embedding, device='cuda')

                        pos_img_path = os.path.join(self.image_dir, label, pos_img)
                        pos_embedding = self._get_embedding(pos_img_path)
                        pos_embedding = torch.tensor(pos_embedding, device='cuda')

                        pos_distance = l2_distance.forward(pos_embedding.unsqueeze(0), anchor_embedding.unsqueeze(0))
                        pos_distance = pos_distance.cpu().item()

                        neg_image = None
                        while neg_image is None:
                            random_neg_labels = random.sample([key for key in grouped.keys() if key != label], 20)
                            neg_img=None
                            for neg_label in random_neg_labels:
                                neg_img = random.choice(grouped[neg_label])
                                neg_img_path = os.path.join(self.image_dir, neg_label, neg_img)
                                neg_embedding = self._get_embedding(neg_img_path)
                                neg_embedding = torch.tensor(neg_embedding, device='cuda')

                                neg_distance = l2_distance.forward(neg_embedding.unsqueeze(0), anchor_embedding.unsqueeze(0))
                                neg_distance = neg_distance.cpu().item()

                                if neg_distance < pos_distance:
                                    acc_hard += 1
                                    neg_image = neg_img
                                    break
                            neg_image = neg_img

                        writer.writerow([anchor_img, pos_img, neg_image])

                        pairs_generated += 1
                        pbar.update(1)

                        if pairs_generated >= num_hard_pairs:
                            break 
            print(f'Hard pairs: {acc_hard}')
            pairs_generated = 0
            with tqdm(total=num_random_pairs, desc="Generating Random Pairs", unit="pair") as pbar:
                while pairs_generated < num_random_pairs:
                    for label, images in grouped.items():
                        if len(images) < 2:
                            continue

                        anchor_img, pos_img = random.sample(images, 2)
                        anchor_img_path = os.path.join(self.image_dir, label, anchor_img)
                        pos_img_path = os.path.join(self.image_dir, label, pos_img)

                        random_neg_label = random.choice([key for key in grouped.keys() if key != label])
                        neg_img = random.choice(grouped[random_neg_label])
                        neg_img_path = os.path.join(self.image_dir, random_neg_label, neg_img)

                        writer.writerow([anchor_img, pos_img, neg_img])

                        pairs_generated += 1
                        pbar.update(1)

                        if pairs_generated >= num_random_pairs:
                            break

        del self.model
        torch.cuda.empty_cache()


    def generate_triplets(self):
        triplets = []

        print("\nGenerating {} triplets ...".format(self.num_triplets))
        if self.epoch % 1 == 0:
            self.gen_triplet_csv()
        data = pd.read_csv(self.triplet_file)


        sampled_triplets = data.sample(n=self.num_triplets, replace=False).values.tolist()
        for triplet in sampled_triplets:
            anchor_img, positive_img, negative_img = triplet
            anchor_id = self.id_to_class[anchor_img]
            negative_id = self.id_to_class[negative_img]
            triplets.append([
                anchor_img,
                positive_img,
                negative_img,
                anchor_id,
                negative_id,
            ])

        return triplets

  
    def __getitem__(self, idx):
        anc_id, pos_id, neg_id, pos_class, neg_class= self.training_triplets[idx]

        anc_img_path = self._find_image_path(anc_id)
        pos_img_path = self._find_image_path(pos_id)
        neg_img_path = self._find_image_path(neg_id)

        anc_img = Image.open(anc_img_path)
        pos_img = Image.open(pos_img_path)
        neg_img = Image.open(neg_img_path)

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,
            'pos_class': pos_class,
            'neg_class': neg_class
        }

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample

    def __len__(self):
        return len(self.training_triplets)



