import os
import pickle
import h5py
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

def compute_stats(input_traj, robot_future, input_goals):
    stats = {}
    stats["input_traj_mean"] = torch.mean(input_traj[:,-1,:], dim=0)
    stats["input_traj_std"] = torch.std(input_traj[:,-1,:], dim=0)
    stats["robot_future_mean"] = torch.mean(robot_future[:,-1,:], dim=0)
    stats["robot_future_std"] = torch.std(robot_future[:,-1,:], dim=0)
    stats["input_goals_mean"] = torch.mean(input_goals, dim=(0,2))
    stats["input_goals_std"] = torch.std(input_goals, dim=(0,2))

    # replace any 0 std with 1
    for key in stats.keys():
        if key[-4:] == "_std":
            if torch.sum(stats[key] == 0) > 0:
                stats[key][stats[key] == 0] = 1

    return stats

def compute_stats_h5(input_traj, robot_future, input_goals):
    stats = {}
    stats["input_traj_mean"] = np.mean(input_traj[:,-1,:], axis=0)
    stats["input_traj_std"] = np.std(input_traj[:,-1,:], axis=0)
    stats["robot_future_mean"] = np.mean(robot_future[:,-1,:], axis=0)
    stats["robot_future_std"] = np.std(robot_future[:,-1,:], axis=0)
    stats["input_goals_mean"] = np.mean(input_goals, axis=(0,2))
    stats["input_goals_std"] = np.std(input_goals, axis=(0,2))

    # replace any 0 std with 1
    for key in stats.keys():
        if key[-4:] == "_std":
            if np.sum(stats[key] == 0) > 0:
                stats[key][stats[key] == 0] = 1

    return stats

class ProbSimTrajDataset(Dataset):
    def __init__(self, path : str, history : int = 5, horizon : int = 5, mode : str = "train", goal_mode : str = "static", stats_file : str = None):
        self.mode = mode
        self.history = history
        self.horizon = horizon

        # get file extension
        ext = path.split(".")[-1]
        if ext == "pkl":
            self.is_h5 = False
            # load pickle file containing processed data
            with open(path, "rb") as f:
                data = pickle.load(f)
            input_traj = data["input_traj"]
            robot_future = data["robot_future"]
            input_goals = data["input_goals"]
            labels = data["labels"]

            # create stats filename
            if stats_file is None:
                stats_file = path.replace(".pkl", "_stats.pkl")
            self.stats_file = stats_file

            if mode == "train":
                stats = compute_stats(input_traj, robot_future, input_goals)
                # save stats
                with open(stats_file, "wb") as f:
                    pickle.dump(stats, f)
            
            # load stats
            with open(stats_file, "rb") as f:
                stats = pickle.load(f)
            # normalize data
            input_traj = (input_traj - stats["input_traj_mean"]) / stats["input_traj_std"]
            robot_future = (robot_future - stats["robot_future_mean"]) / stats["robot_future_std"]
            input_goals = (input_goals.transpose(1,2) - stats["input_goals_mean"]) / stats["input_goals_std"]
            input_goals = input_goals.transpose(1,2)

            self.input_traj = input_traj
            self.robot_future = robot_future
            self.input_goals = input_goals
            self.labels = labels
            self.length = len(input_traj)
        elif ext == "h5":
            self.is_h5 = True
            self.path = path
            with h5py.File(path, "r") as f:
                input_traj = f["input_traj"]
                robot_future = f["robot_future"]
                input_goals = f["input_goals"]

                # compute stats if filename doesn't exist
                if stats_file is None:
                    stats_file = path.replace(".h5", "_stats.pkl")
                self.stats_file = stats_file

                # check if this stats file exists
                if not os.path.exists(stats_file):
                    stats = compute_stats_h5(input_traj, robot_future, input_goals)
                    # save stats
                    with open(stats_file, "wb") as f:
                        pickle.dump(stats, f)
                
                # load stats
                with open(stats_file, "rb") as f:
                    stats = pickle.load(f)
                self.stats = stats
                self.length = len(input_traj)

        #         if mode == "train":
        #             self.length = int(0.8 * len(input_traj))
        #             self.offset = 0
        #         elif mode == "val":
        #             self.length = int(0.2 * len(input_traj))
        #             self.offset = int(0.8 * len(input_traj))

        # if mode == "train":
        #     self.length = int(0.8 * len(input_traj))
        #     self.offset = 0
        # elif mode == "val":
        #     self.length = int(0.2 * len(input_traj))
        #     self.offset = int(0.8 * len(input_traj))

    def open_h5(self):
        data = h5py.File(self.path, "r")
        self.input_traj = data["input_traj"]
        self.robot_future = data["robot_future"]
        self.input_goals = data["input_goals"]
        self.labels = data["labels"]

    def __getitem__(self, index):
        if not self.is_h5:
            return (self.input_traj[index], self.robot_future[index], self.input_goals[index]), self.labels[index]
        
        if not hasattr(self, 'input_traj'):
            self.open_h5()
        # idx = (index % self.length) + self.offset
        idx = index

        # normalize data point
        # TODO: convert means and SDs to torch tensors for speed
        input_traj = (self.input_traj[idx] - self.stats["input_traj_mean"]) / self.stats["input_traj_std"]
        robot_future = (self.robot_future[idx] - self.stats["robot_future_mean"]) / self.stats["robot_future_std"]
        input_goals = (self.input_goals[idx].T - self.stats["input_goals_mean"]) / self.stats["input_goals_std"]
        input_goals = input_goals.T

        return (torch.tensor(input_traj).float(), torch.tensor(robot_future).float(), torch.tensor(input_goals).float()), torch.tensor(self.labels[idx]).float()

    def __len__(self):
        return self.length
