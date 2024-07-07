import torch
import numpy as np
import os
import torchvision.transforms as transforms
from FaceRecognition.main import FaceNetModel
from FaceRecognition.utils.config import opt
import os
from sklearn.neighbors import KNeighborsClassifier



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

FaceNet_model = FaceNetModel
data_transforms = transforms.Compose([
    transforms.Resize(size=opt.target_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.6071, 0.4609, 0.3944],
        std=[0.2457, 0.2175, 0.2129]
    )
])

def get_embedding(image):
    image = data_transforms(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        embedding = FaceNet_model(image).cpu().numpy().flatten()
    return embedding


def recognize_faces(faces, embeddings_dir='../embeddings4', k=10):
    face_embeddings = []
    for face in faces:
        embedding = get_embedding(face)
        face_embeddings.append(embedding)
    
    face_embeddings = np.array(face_embeddings).astype('float32')
    
    identity_files = [f for f in os.listdir(embeddings_dir) if f.endswith('.npy')]
    identity_embeddings = {}
    identity_labels = []

    for identity_file in identity_files:
        identity = os.path.splitext(identity_file)[0]
        embeddings = np.load(os.path.join(embeddings_dir, identity_file))
        identity_embeddings[identity] = embeddings
        identity_labels.extend([identity] * len(embeddings))
    
    all_identity_embeddings = np.vstack(list(identity_embeddings.values()))
    all_identity_labels = np.array(identity_labels)

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(all_identity_embeddings, all_identity_labels)

    distances, indices = knn.kneighbors(face_embeddings, n_neighbors=k)
    predicted_ids = knn.predict(face_embeddings)
    
    best_matches = []
    for i, dist_list in enumerate(distances):
        close_neighbors = [dist for dist in dist_list if dist < 0.9]
        if len(close_neighbors) >= 5:
            best_matches.append(predicted_ids[i])
        else:
            best_matches.append("") 
    
    best_distances = distances.tolist()


    return best_matches, best_distances
