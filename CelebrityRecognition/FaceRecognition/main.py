import torch
from FaceRecognition.model.NNInception import model


pretrain_path = 'CelebrityRecognition/FaceRecognition/model_training_checkpoints/model_pretrained.pt'
checkpoint = torch.load(pretrain_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
FaceNetModel= model