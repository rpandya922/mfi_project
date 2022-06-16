import numpy as np
import torch
from torch.utils.data import DataLoader

from intention_predictor import create_model, IntentionPredictor
from dataset import SimTrajDataset

if __name__ == "__main__":
    # load npz dataset file
    traj_data = np.load("./data/simulated_interactions.npz")
    dataset = SimTrajDataset(traj_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictor = create_model()

    for batch_idx, (data, target) in enumerate(loader):
        predictor(*data)
