import numpy as np
import matplotlib.pyplot as plt

from human import Human, DIDynamics


if __name__ == "__main__":
    ts = 0.1
    x0 = np.zeros((4, 1))
    goal = np.zeros_like(x0)
    goal[0,0] = 5.0
    goal[2,0] = 5.0

    dynamics = DIDynamics(ts)
    human = Human(x0, dynamics, goal)

    fig, ax = plt.subplots()

    for t in range(100):
        u = human.get_u()
        x = human.step(u)

        ax.cla()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.scatter(goal[0], goal[2])
        ax.scatter(x[0], x[2])

        plt.pause(0.01)
    # plt.show()

