import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    filename = "./data/sim_stats_20230802-171349.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    baseline = data["baseline"]
    multimodal = data["multimodal"]
    SEA = data["SEA"]

    for i in range(10):
        fig, ax = plt.subplots()
        ax.plot(baseline[i][5], label="baseline")
        ax.plot(multimodal[i][5], label="multimodal")
        ax.plot(SEA[i][5], label="SEA")
        ax.plot(np.ones_like(baseline[i][5]), c="black", linestyle="--", label="min distance")
        ax.set_xlabel("timestep")
        ax.set_ylabel("distance b/w agents")
        ax.legend()
        plt.show()