import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    filename = "./data/sim_stats_20230803-162403.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)

    baseline = data["baseline"]
    multimodal = data["multimodal"]
    SEA = data["SEA"]

    goals_reached = {"baseline": [], "multimodal": [], "SEA": []}
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes = axes.flatten()
    for i in range(100):
        
        ax = axes[0]
        ax.clear()
        ax.plot(baseline[i]["distances"], label="baseline")
        ax.plot(multimodal[i]["distances"], label="multimodal")
        ax.plot(SEA[i]["distances"], label="SEA")
        ax.plot(np.ones_like(baseline[i]["distances"]), c="black", linestyle="--", label="min distance")
        ax.set_xlabel("timestep")
        ax.set_ylabel("distance b/w agents")
        ax.legend()

        # add a verical line for each time the human reaches a goal (SEA only)
        for j, r_goal_reached in enumerate(SEA[i]["h_goal_reached"]):
            if r_goal_reached >= 0:
                ax.axvline(x=j, c="red", linestyle="--", alpha=0.5)

        ax = axes[1]
        ax.clear()
        ax.plot(baseline[i]["r_goal_dists"], label="baseline")
        ax.plot(multimodal[i]["r_goal_dists"], label="multimodal")
        ax.plot(SEA[i]["r_goal_dists"], label="SEA")
        ax.set_xlabel("timestep")
        ax.set_ylabel("distance to goal")

        baseline_goals = (np.array(baseline[i]["r_goal_reached"]) >= 0).sum()
        multimodal_goals = (np.array(multimodal[i]["r_goal_reached"]) >= 0).sum()
        SEA_goals = (np.array(SEA[i]["r_goal_reached"]) >= 0).sum()
        goals_reached["baseline"].append(baseline_goals)
        goals_reached["multimodal"].append(multimodal_goals)
        goals_reached["SEA"].append(SEA_goals)
        print(f"baseline: {baseline_goals}, multimodal: {multimodal_goals}, SEA: {SEA_goals}")

        plt.pause(0.001)
        input(": ")
    print(np.mean(goals_reached["baseline"]))
    print(np.mean(goals_reached["multimodal"]))
    print(np.mean(goals_reached["SEA"]))