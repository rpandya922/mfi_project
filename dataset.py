from torch.utils.data.dataset import Dataset

class SimTrajDataset(Dataset):
    def __init__(self, data, mode : str = "train"):
        self.mode = mode

        # TODO: process trajectories into dataset
