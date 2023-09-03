import os
import time
import numpy as np
import pickle
from random import shuffle, randint

from test_bayes_aruco import bayes_inf_rs2, practice_round

def run_n_games(folder, robot_type, n_games, user=-1):
    for game_idx in range(n_games):
        np.random.seed(game_idx) # so we can do an exact comparison for the first goal selection of each robot type
        data = bayes_inf_rs2(robot_type=robot_type, mode="study", round_num=game_idx, user=user)
        # save this trial's data to pkl file
        filename = f"{robot_type}_trial{game_idx}.pkl"
        filepath = folder + filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        if game_idx != n_games-1:
            input("Hit [Enter] to continue to next round: ")


if __name__ == "__main__":
    # user = "test"
    user = str(randint(1000, 9999))
    print(f"User: {user}")
    controllers = ["baseline", "baseline_belief", "cbp"]
    robot_colors = {"baseline": "brown", "baseline_belief": "purple", "cbp": "green"}
    shuffle(controllers)

    n_games = 4 # number of games to play with each controller
    # make folder for this user (+ timestamp)
    foldername = f"./data/user_study/user_{user}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(os.path.dirname(foldername), exist_ok=True)
    filepath = f"{foldername}/robot_order.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(controllers, f)

    # run practice round
    input("Hit [Enter] to play practice round")
    data = practice_round(mode="study")
    filename = "practice_round.pkl"
    filepath = foldername + filename
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print()

    # run n_games for each controller
    for robot_idx, robot in enumerate(controllers):
        input(f"Hit [Enter] to start games with robot {robot_idx}.")
        run_n_games(foldername, robot, n_games, user)
        input(f"Games with robot {robot_idx} done. Please fill out the survey. Hit [Enter] to continue.")
        print()
    # print robot color order
    for controller in controllers:
        print(robot_colors[controller])
