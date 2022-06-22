import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from intention_predictor import create_model, IntentionPredictor
from dataset import SimTrajDataset

def train(model, optimizer, trainset_loader, valset_loader, epoch=50):
    all_train_loss = []
    all_val_loss = []
    batch_size = trainset_loader.batch_size

    iteration = 0
    for ep in tqdm(range(epoch)):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # forward pass
            model_out = model(*data)

            # compute loss
            loss = nn.CrossEntropyLoss(reduction="sum")
            output = loss(model_out, target)
            total_loss += output.item()

            output.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration += 1
        all_train_loss.append(total_loss / (batch_idx+1) / batch_size)

        # test on validation
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(valset_loader):
            model_out = model(*data)

            val_loss += loss(model_out, target).item()
        all_val_loss.append(val_loss / (batch_idx+1) / batch_size)

    return all_train_loss, all_val_loss

if __name__ == "__main__":
    # load npz dataset file
    traj_data = np.load("./data/simulated_interactions.npz")
    dataset = SimTrajDataset(traj_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # validation data
    traj_data = np.load("./data/simulated_interactions2.npz")
    val_dataset = SimTrajDataset(traj_data)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictor = create_model()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=20)

    # save model
    torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor.pt")

    plt.plot(all_train_loss, label="train")
    plt.plot(all_val_loss, label="val")
    plt.yscale("log")
    plt.legend()
    plt.show()
