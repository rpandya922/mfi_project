import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class SimTrajDataset(Dataset):
    def __init__(self, data, history : int = 5, horizon : int = 5, mode : str = "train", goal_mode : str = "static"):
        self.mode = mode
        self.history = history
        self.horizon = horizon

        # process trajectories into dataset
        xr_traj = data["xr_traj"]
        xh_traj = data["xh_traj"]

        goals = data["goals"]
        chosen_goal_idx = data["goal_idx"]

        # process data into chunks of length `history`
        traj_len, n_traj = xr_traj.shape[1:]

        input_traj = []
        robot_future = []
        input_goals = []
        labels = []

        for i in range(n_traj):
            for j in range(history, traj_len - horizon):
                # TODO: zero-pad instead of skipping
                xh_hist = xh_traj[:,j-history:j,i]
                xr_hist = xr_traj[:,j-history:j,i]
                xr_future = xr_traj[:,j:j+horizon,i]

                # LSTM expects input of size (sequence length, # features) [batch size dealth with separately]
                input_traj.append(torch.tensor(np.vstack((xh_hist, xr_hist)).T).float().to(device)) # shape (5,8)
                robot_future.append(torch.tensor(xr_future.T).float().to(device)) # shape (5,4)
                if goal_mode == "static":
                    input_goals.append(torch.tensor(goals[:,:,i]).float().to(device)) # shape (4,3)
                    labels.append(torch.tensor(chosen_goal_idx[i]).to(device))
                elif goal_mode == "dynamic":
                    traj_goals = goals[0:4,:,:,i]
                    goals_hist = traj_goals[:,:,j-history:j].reshape((traj_goals.shape[0]*traj_goals.shape[1],history))
                    input_goals.append(torch.tensor(goals_hist.T).float().to(device)) # shape (5,12)
                    labels.append(torch.tensor(chosen_goal_idx[j,i]).to(torch.int64).to(device))

        self.input_traj = input_traj
        self.robot_future = robot_future
        self.input_goals = input_goals
        self.labels = labels

    def __getitem__(self, index):
        return (self.input_traj[index], self.robot_future[index], self.input_goals[index]), self.labels[index]

    def __len__(self):
        return len(self.input_traj)

class ProbSimTrajDataset(Dataset):
    def __init__(self, path : str, history : int = 5, horizon : int = 5, mode : str = "train", goal_mode : str = "static"):
        self.mode = mode
        self.history = history
        self.horizon = horizon

        # load pickle file containing processed data
        with open(path, "rb") as f:
            data = pickle.load(f)

        input_traj = data["input_traj"]
        robot_future = data["robot_future"]
        input_goals = data["input_goals"]
        labels = data["labels"]

        self.input_traj = input_traj
        self.robot_future = robot_future
        self.input_goals = input_goals
        self.labels = labels

    def __getitem__(self, index):
        return (self.input_traj[index], self.robot_future[index], self.input_goals[index]), self.labels[index]

    def __len__(self):
        return len(self.input_traj)
