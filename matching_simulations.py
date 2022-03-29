""" Simulate the value function and the payments in the two-by-two matching game """

import cmd, sys
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from src.convergence import converge


if __name__ == "__main__":

    power = int(
        input(
            "Input an integer. The resulting grid will have 2 ** (your number) points. Recommended greater than 6"
        )
    )  # recommended value greater than 6
    epsilon = float(
        input(
            "Input a float epsilon - a threshold for convergence. Recommended lower than 0.01"
        )
    )  # recommended lower than 0.01
    # power = 6
    # epsilon = 0.005

    N_grid = 2**power  # number of grid points
    N_times = N_grid + 1

    STOP = 10000

    p_direct = [i / 10 for i in range(0, 11)]  # probability of a (direct) rank matching

    # types of players
    man_type = np.arange(N_times) / N_grid

    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    axs = fig.subplots(4, 3).flatten()
    value_star = np.divide(
        np.sqrt(man_type), (np.sqrt(man_type) + np.sqrt(1 - man_type))
    )

    for ax, p in zip(axs, p_direct):
        value_fn = np.zeros(N_times)

        # Seed for a value function
        # value_fn = uniform(low=0, high = 1, size = N_times) # random seed
        value_fn = np.repeat(1, repeats=N_times)  # deterministic seed

        counter = 0

        value_fn = converge(value_fn, p, epsilon, power, counter, STOP)

        ax.set_title("Prob of rank-match: %s" % str(p))
        ax.plot(man_type, value_fn, "o", ls="-", ms=4)
        ax.plot(man_type, value_star, color="red")

    plt.savefig("value_functions.pdf", format="pdf")
    plt.close("all")
