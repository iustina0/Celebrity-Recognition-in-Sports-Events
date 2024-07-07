
import torchvision.transforms as transforms

class FacenetConfig:
    image_dir = '../../images'
    identity_file = '../../list_identity_celeba.txt'
    triplet_file = 'FaceRecognition/data/csvs/train_triplets.csv'
    parsed_identity_file = 'FaceRecognition/data/csvs/output.csv'

    env = 'facenet'
    target_size = (199, 199)

    epoch = 14
    batch_size = 32

    test_pairs_num = 1000

    train_transform = transforms.Compose([
        transforms.Resize(size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(size=target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])


opt = FacenetConfig()
