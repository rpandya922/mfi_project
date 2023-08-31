import os
import time
import numpy as np
import pickle

from test_bayes_aruco import bayes_inf_rs2

if __name__ == "__main__":
    user = "test"
    controllers = ["baseline", "baseline_belief", "cbp"]
    robot_colors = {"baseline": "brown", "baseline_belief": "purple", "cbp": "green"}
    n_games = 5 # number of games to play with each controller
    # make folder for this user (+ timestamp)
    foldername = f"./data/user_study/user_{user}_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(os.path.dirname(foldername), exist_ok=True)

    # TODO: run n_games for each controller (and randomize the order of the controllers)
    # TODO: make each robot a different color (consistent color => go to survey for that color robot)
