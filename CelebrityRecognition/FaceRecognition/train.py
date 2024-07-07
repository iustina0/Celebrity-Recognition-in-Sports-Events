import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm
import numpy as np
from data.evaldata import EvalDataset
from utils.config import opt
from data.traindata import TripletFaceDataset
from model.NN2Inception import NN2Inception #, NN2Inception
from FaceRecognition.model.incept import Resnet34Triplet 
from utils.eval_tools import *
from utils.plot_tools import *
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = 0.2
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist(anchor, positive)
        neg_dist = self.pdist(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)

        loss = torch.mean(hinge_dist)
        return loss


def forward_pass(images, model, batch_size):
    embeddings = model(images)
    anc_embeddings = embeddings[:batch_size]
    pos_embeddings = embeddings[batch_size:batch_size*2]
    neg_embeddings = embeddings[batch_size*2:]
    return anc_embeddings, pos_embeddings, neg_embeddings



def evaluate(model, dataloader, epoch, device='cuda'):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2).to(device)
        distances, labels = [], []

        print("Validating! ...")
        progress_bar = enumerate(tqdm(dataloader))
        for _, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.concatenate(labels)
        distances = np.concatenate(distances)
        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, tar, far, best_threshold = evaluate_data(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )

        print(f'Mean positive distance: {np.mean(distances[labels == 1])}   Mean negative distance: {np.mean(distances[labels == 0])}')
        log_message = (
            f"Accuracy: {np.mean(accuracy)}\tPrecision {np.mean(precision)}\tRecall {np.mean(recall)}\tROC Area Under Curve: {roc_auc}\tBest distance threshold: {best_threshold}\tTAR: {np.mean(tar)} @ FAR: {np.mean(far)}\n"
        )
        print(log_message)
    try:
        plot_roc(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name=f"FaceRecognition/plots/roc_plot.png"
        )
    except Exception as e:
        print(e)
    return best_distances


def train(model, optimizer_model, epochs=14, batch_size=32, num_workers=4, margin=0.2, device='cuda', start_epoch=0):
    dataloader = torch.utils.data.DataLoader(
        dataset=EvalDataset(
            transform=opt.eval_transform
        ),
        batch_size=32,
        num_workers=num_workers,
        shuffle=False
    )
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, start_epoch + epochs):
        print(optimizer_model.param_groups[0]['lr'])
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(p=2).to(device)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletFaceDataset(
                epoch=epoch,
                transform=opt.train_transform,
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False 
        )

        model.train()
        num_valid_training_triplets = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for _, batch_sample in progress_bar:
            anc_imgs = batch_sample['anc_img'].to(device)
            pos_imgs = batch_sample['pos_img'].to(device)
            neg_imgs = batch_sample['neg_img'].to(device)

            all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))

            anc_embeddings, pos_embeddings, neg_embeddings = forward_pass(all_imgs, model, batch_size)
            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)
            all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
            valid_triplets = np.where(all == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets]
            pos_valid_embeddings = pos_embeddings[valid_triplets]
            neg_valid_embeddings = neg_embeddings[valid_triplets]

            triplet_loss = TripletLoss(margin=margin).to(device)(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
            )

            num_valid_training_triplets += len(anc_valid_embeddings)

            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

        print(f'Epoch {epoch}:\tNumber of valid training triplets in epoch: {num_valid_training_triplets}')

        model.eval()
        val_loss = evaluate(model, dataloader, epoch, device)
        model.train()
        print(f'Epoch {epoch}:\tValidation Loss: {val_loss}')

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'val_loss': val_loss
        }

        torch.save(state, f'model_training_checkpoints/model_triplet_epoch_{epoch}.pt')

def resume(model, optimizer_model, path, epochs=14):
    resume_path = path
    if os.path.isfile(resume_path):
        print("Loading checkpoint {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        start_epoch =  checkpoint['epoch'] + 1

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
        print(f"Checkpoint loaded: start epoch from checkpoint = {start_epoch}")
        
        optimizer_model.param_groups[0]['lr'] = 0.008
        train(model, optimizer_model, start_epoch=start_epoch, epochs=epochs)
    else:
        print(f"No checkpoint found at {resume_path}!")



if __name__ == "__main__":
    model = NN2Inception().cuda()
    optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=0.04,
            lr_decay=0.001,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=1e-10
        )
    train(model, optimizer_model, epochs=40)

    # resume(model, optimizer_model, path=f'model_training_checkpoints/model_triplet_epoch_{920}.pt', epochs=500)