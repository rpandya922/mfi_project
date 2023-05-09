import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from intention_predictor import create_model, IntentionPredictor
from dataset import SimTrajDataset, ProbSimTrajDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Training with device {device}")

def train(model, optimizer, trainset_loader, valset_loader, epoch=50):
    all_train_loss = []
    all_val_loss = []
    batch_size = trainset_loader.batch_size

    writer = SummaryWriter(log_dir=os.path.join(".", "data", "prob_pred", "runs", time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + f"_lr_{optimizer.param_groups[0]['lr']}_bs_{batch_size}"))

    loss = nn.CrossEntropyLoss(reduction="mean").to(device)

    iteration = 0
    for ep in tqdm(range(epoch)):

        # test on validation
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(valset_loader):
            # send data to device
            data = [d.to(device) for d in data]
            target = target.to(device)
            
            model_out = model(*data)

            val_loss += loss(model_out, target).item()
        all_val_loss.append(val_loss / (batch_idx+1))
        writer.add_scalar("Loss/val", all_val_loss[-1], ep)

        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(trainset_loader):
            # send data to device
            data = [d.to(device) for d in data]
            target = target.to(device)

            # zero gradients
            optimizer.zero_grad()

            # forward pass
            model_out = model(*data)

            # compute loss
            output = loss(model_out, target)
            total_loss += output.item()

            output.backward()
            optimizer.step()

            iteration += 1
        all_train_loss.append(total_loss / (batch_idx+1))
        writer.add_scalar("Loss/train", all_train_loss[-1], ep)

    writer.flush()
    writer.close()
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
    # traj_data = np.load("./data/simulated_interactions_bayes.npz")
    # version that makes human go to least-likely robot goal (ll = least likely)
    traj_data = np.load("./data/simulated_interactions_bayes_ll.npz")
    # traj_data = np.load("./data/simulated_interactions_rule.npz")
    dataset = SimTrajDataset(traj_data, horizon=horizon)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # validation data
    # traj_data = np.load("./data/simulated_interactions2.npz")
    # traj_data = np.load("./data/simulated_interactions_bayes2.npz")
    traj_data = np.load("./data/simulated_interactions_bayes_ll2.npz")
    # traj_data = np.load("./data/simulated_interactions_rule2.npz")
    val_dataset = SimTrajDataset(traj_data, horizon=horizon)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    predictor = create_model(horizon_len=horizon)
    # predictor = FC()
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=50)

    # save model
    # TODO: don't overwrite existing model, save into new file based on date/time
    # torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_plan20.pt")
    # torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_bayes.pt")
    torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_bayes_ll.pt")
    # torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_rule.pt")

    plt.plot(all_train_loss, label="train")
    plt.plot(all_val_loss, label="val")
    plt.yscale("log")
    plt.legend()
    # plt.show()
    plt.savefig("./data/train_loss.png")

def train_sim_noplan():
    horizon = 20
    # load npz dataset file
    traj_data = np.load("./data/simulated_interactions_bayes.npz")
    dataset = SimTrajDataset(traj_data, horizon=horizon)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # validation data
    traj_data = np.load("./data/simulated_interactions_bayes2.npz")
    val_dataset = SimTrajDataset(traj_data, horizon=horizon)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    predictor = create_model(horizon_len=horizon, use_plan=False)
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=50)

    # save model
    # TODO: don't overwrite existing model, save into new file based on date/time
    torch.save(predictor.state_dict(), "./data/models/sim_intention_predictor_bayes_noplan.pt")

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
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

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

def train_prob_sim(save_model=True):
    horizon = 20
    # load datasets
    train_path = "./data/prob_pred/simulated_interactions_bayes_prob_train2_processed.pkl"
    dataset = ProbSimTrajDataset(path=train_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # validation data
    val_path = "./data/prob_pred/simulated_interactions_bayes_prob_val2_processed.pkl"
    val_dataset = ProbSimTrajDataset(path=val_path)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    predictor = create_model(horizon_len=horizon)
    predictor = predictor.to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=4e-3)
    all_train_loss, all_val_loss = train(predictor, optimizer, loader, val_loader, epoch=45)

    if save_model:
        # save model into new file based on date/time
        model_name = "./data/models/prob_pred_intention_predictor_bayes_{}.pt".format(time.strftime("%Y%m%d-%H%M%S"))
        # create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        torch.save(predictor.state_dict(), model_name)

    plt.plot(all_train_loss, label="train")
    plt.plot(all_val_loss, label="val")
    # plt.yscale("log")
    plt.legend()
    # plt.show()
    plt.savefig("./data/train_loss.png")
    plt.yscale("log")
    plt.savefig("./data/train_loss_log.png")

def split_prob_train_data():
    import pickle
    train_path = "./data/prob_pred/simulated_interactions_bayes_prob_train_processed.pkl"
    # load training data
    with open(train_path, "rb") as f:
        data = pickle.load(f)
    input_traj = data["input_traj"]
    robot_future = data["robot_future"]
    input_goals = data["input_goals"]
    labels = data["labels"]

    # split into train and val
    train_input_traj = input_traj[:int(len(input_traj)*0.8)]
    train_robot_future = robot_future[:int(len(robot_future)*0.8)]
    train_input_goals = input_goals[:int(len(input_goals)*0.8)]
    train_labels = labels[:int(len(labels)*0.8)]

    val_input_traj = input_traj[int(len(input_traj)*0.8):]
    val_robot_future = robot_future[int(len(robot_future)*0.8):]
    val_input_goals = input_goals[int(len(input_goals)*0.8):]
    val_labels = labels[int(len(labels)*0.8):]

    # save into new file
    train_path_new = "./data/prob_pred/simulated_interactions_bayes_prob_train_processed_new.pkl"
    val_path_new = "./data/prob_pred/simulated_interactions_bayes_prob_val_processed_new.pkl"

    with open(train_path_new, "wb") as f:
        pickle.dump({
            "input_traj": train_input_traj,
            "robot_future": train_robot_future,
            "input_goals": train_input_goals,
            "labels": train_labels
        }, f)

    with open(val_path_new, "wb") as f:
        pickle.dump({
            "input_traj": val_input_traj,
            "robot_future": val_robot_future,
            "input_goals": val_input_goals,
            "labels": val_labels
        }, f)

if __name__ == "__main__":
    # train_bis_sim()
    # train_sim()
    # train_sim_noplan()
    train_prob_sim()
