import numpy as np
import matplotlib.pyplot as plt
import pickle

def process_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    robot_types = ["cbp", "baseline", "baseline_belief"]
    for robot_type in robot_types:
        robot_data = data[robot_type]
        print(robot_type)
        for idx, trial_data in enumerate(robot_data):
            h_goal_idxs = trial_data["h_goal_idxs"]
            # print(h_goal_idxs[0] == h_goal_idxs[-1], len(h_goal_idxs))
            # compute number of times human's goal changed
            diffs = np.diff(h_goal_idxs, n=1)
            num_goal_changes = np.count_nonzero(diffs)
            print(num_goal_changes)
        print()

if __name__ == "__main__":
    filepath = "./data/cbp_sim/cbp_compare_20230817-175956.pkl"
    process_data(filepath)

        