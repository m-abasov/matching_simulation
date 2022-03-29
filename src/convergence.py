import numpy as np
from src.integration_nonconvex import meanVal


def converge(value_fn, p, epsilon, power, counter, STOP):
    """Implements recursive convergence of the value function.

    Inputs:
    - value_fn: a coandidate value function
    - p: the probability of rank-match
    - epsilon: threshold for convergence,
    - power: determines the number of points on the grid
    - counter: counts the number of recursions
    - STOP: max value of recursions allowed

    Outputs: next iteration of the candidate value function OR the final estimate of the value function.

    """

    N_times = len(value_fn)
    N_grid = N_times - 1

    # all possible types of the players
    man_type = np.arange(N_times) / N_grid
    woman_type = np.arange(N_times) / N_grid

    value_prime = np.zeros(N_times)

    # Compute best responses of each type of the player for a given candidate value function
    for i in range(N_times):
        j_low = 0
        j_high = N_times - 1
        for itr in range(power):
            j_mid = int((j_low + j_high) / 2)
            woman_bar = meanVal(man_type[i], woman_type[j_mid], value_fn, p, N_times)
            if woman_bar > woman_type[j_mid]:
                j_low = j_mid
            else:
                j_high = j_mid
        delta_low = meanVal(man_type[i], woman_type[j_low], value_fn, p, N_times)
        delta_high = meanVal(man_type[i], woman_type[j_high], value_fn, p, N_times)
        if delta_high > delta_low:
            value_prime[i] = woman_type[j_low]
        else:
            value_prime[i] = woman_type[j_high]

    delta = np.max(np.absolute(value_fn - value_prime))

    # Check if convergence is achieved
    if delta < epsilon:
        return value_fn

    # Check if number of recursions is not too high
    if counter >= STOP:
        print("Recursion maxed out")
        return

    # Recursive step
    return converge(value_prime, p, epsilon, power, counter + 1, STOP)
