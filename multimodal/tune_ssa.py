import os
import numpy as np
from tqdm import tqdm
import pickle

from simulate import run_trajectory

if __name__ == "__main__":
    n_sim = 10
    controller = "baseline"
    np.random.seed(0)
    etas = np.linspace(0, 30, 30)
    k_phis = np.linspace(0, 30, 30)
    dmin = 3.0 

    res = {}
    min_coll = np.inf
    for eta in tqdm(etas):
        for k_phi in k_phis:
            collisions = 0
            for i in range(n_sim):
                res = run_trajectory(controller=controller, plot=False, n_goals=3, eta=eta, k_phi=k_phi)
                collisions += np.sum(np.array(res["distances"]) < dmin)
            if collisions < min_coll:
                min_coll = collisions
                print(f"eta: {eta}, k_phi: {k_phi}, collisions: {collisions}")
            res[(eta, k_phi)] = collisions
    # save file
    filename = f"./data/tune_ssa_{controller}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(res, f)