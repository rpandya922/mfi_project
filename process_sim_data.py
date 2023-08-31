import numpy as np
import matplotlib.pyplot as plt
import pickle

def process_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    robot_types = ["cbp", "baseline", "baseline_belief", "cbp_nn"]
    # robot_types = ["cbp_nn"]
    all_stats = {robot_type: {} for robot_type in robot_types}
    for robot_type in robot_types:
        robot_data = data[robot_type]
        print(robot_type)
        n_changed = 0
        all_goal_changes = []
        traj_lens = []
        for idx, trial_data in enumerate(robot_data):
            h_goal_idxs = trial_data["h_goal_idxs"]
            # compute number of times human's goal changed
            diffs = np.diff(h_goal_idxs, n=1)
            num_goal_changes = np.count_nonzero(diffs)
            change_idx = np.where(diffs != 0)[0]
            if len(change_idx) > 0:
                n_changed += 1
            traj_lens.append(len(h_goal_idxs))
            all_goal_changes.append(num_goal_changes)
            print(num_goal_changes, change_idx)
        print(n_changed / len(robot_data))
        print(np.mean(all_goal_changes), np.std(all_goal_changes))
        print(np.mean(traj_lens), np.std(traj_lens))
        print()
        all_stats[robot_type]["num_goal_changes"] = all_goal_changes
    
    fig, ax = plt.subplots()
    ax.boxplot([all_stats[robot_type]["num_goal_changes"] for robot_type in robot_types], showmeans=True)
    ax.set_xticklabels(robot_types)
    ax.set_ylabel("Number of Goal Changes")
    plt.show()

def process_full_game_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    robot_types = ["cbp", "baseline", "baseline_belief"]
    # robot_types = ["cbp_nn"]
    all_stats = {robot_type: {} for robot_type in robot_types}
    for robot_type in robot_types:
        robot_data = data[robot_type]
        print(robot_type)
        scores = []
        for idx, trial_data in enumerate(robot_data):
            if idx % 2 == 1:
                print(trial_data["team_score"])
                scores.append(trial_data["team_score"])
        print(np.mean(scores), np.std(scores))
        print()
        all_stats[robot_type]["scores"] = scores
 
    fig, ax = plt.subplots()
    ax.boxplot([all_stats[robot_type]["scores"] for robot_type in robot_types], showmeans=True)
    ax.set_xticklabels(robot_types)
    ax.set_ylabel("Team Score")
    plt.show()


if __name__ == "__main__":
    # filepath = "./data/cbp_sim/cbp_compare_20230821-105615.pkl" # large results file with 1000 trajectories per controller (5 goals)
    # filepath = "./data/cbp_sim/cbp_compare_20230822-104542.pkl" # testing 3 goals again
    # filepath = "./data/cbp_sim/cbp_compare_20230822-130153.pkl" # testing w/ NN CBP too (small), bug in cbp_nn code
    # filepath = "./data/cbp_sim/cbp_compare_20230822-134801.pkl" # large results with NN CBP (3 goals), bug in code
    # filepath = "./data/cbp_sim/cbp_compare_20230823-122447.pkl" # testing no safety
    # process_data(filepath)

    filepath = "./data/cbp_sim/cbp_full_game_20230831-135807.pkl"
    process_full_game_data(filepath)
