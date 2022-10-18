import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from intention_predictor import create_model, IntentionPredictor
from dataset import SimTrajDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Training with device {device}")

def train(model, optimizer, trainset_loader, valset_loader, epoch=50):
    all_train_loss = []
    all_val_loss = []
    batch_size = trainset_loader.batch_size

    loss = nn.CrossEntropyLoss(reduction="sum").to(device)

    iteration = 0
    for ep in tqdm(range(epoch)):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # forward pass
            model_out = model(*data)

            # compute loss
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

class FC(nn.Module):
    def __init__(self, in_dim=72):
        super(FC, self).__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3):
        x = torch.cat((x1.flatten(start_dim=1), x2.flatten(start_dim=1), x3.flatten(start_dim=1)), dim=1)
        return self.fc2(self.relu(self.fc1(x)))

def train_sim():
    horizon = 20
    # load npz dataset file
    # traj_data = np.load("./data/simulated_interactions.npz")
    traj_data = np.load("./data/simulated_interactions_bayes.npz")
    # traj_data = np.load("./data/simulated_interactions_rule.npz")
    dataset = SimTrajDataset(traj_data, horizon=horizon)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # validation data
    # traj_data = np.load("./data/simulated_interactions2.npz")
    traj_data = np.load("./data/simulated_interactions_bayes2.npz")
    # traj_data = np.load("./data/simulated_interactions_rule2.npz")
    val_dataset = SimTrajDataset(traj_data, horizon=horizon)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    predictor = create_model(horizon_len=horizon)
    # predictor = FC()
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=50)

    # save model
    # TODO: don't overwrite existing model, save into new file based on date/time
    # torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_plan20.pt")
    torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_bayes.pt")
    # torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_rule.pt")

    plt.plot(all_train_loss, label="train")
    plt.plot(all_val_loss, label="val")
    plt.yscale("log")
    plt.legend()
    # plt.show()
    plt.savefig("./data/train_loss.png")

def train_bis_sim():
    horizon = 20
    # load npz dataset file
    traj_data = np.load("./data/BIS/simulated_interactions_train.npz")
    dataset = SimTrajDataset(traj_data, horizon=horizon, goal_mode="dynamic")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # validation data
    # TODO: use a different validation set
    traj_data = np.load("./data/BIS/simulated_interactions_test.npz")
    val_dataset = SimTrajDataset(traj_data, horizon=horizon, goal_mode="dynamic")
    val_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    predictor = create_model(horizon_len=horizon, goal_mode="dynamic")
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=50)

    # save model
    # TODO: don't overwrite existing model, save into new file based on date/time
    torch.save(predictor.state_dict(), "./data/models/bis_intention_predictor.pt")

    plt.plot(all_train_loss, label="train")
    plt.plot(all_val_loss, label="val")
    plt.yscale("log")
    plt.legend()
    # plt.show()
    plt.savefig("./data/train_loss.png")

if __name__ == "__main__":
    train_bis_sim()
    # train_sim()
