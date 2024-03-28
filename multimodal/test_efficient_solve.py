from simulate import run_trajectory
from tqdm import tqdm

if __name__ == "__main__":
    for i in tqdm(range(10)):
        controller = "multimodal"
        res = run_trajectory(controller=controller, plot=False, n_goals=4)