""" Legend:
    m_accept - grid matrix with 1s for each grid point for which the match is accepted
    m_nonaccept  - grid matrix with 0s for each grid point for which the match is accepted
    m_direct, m_reversed - partitioning of the grid into (nonequal) quadrants
"""
import numpy as np


def type2index(player_type, N_grid):
    """Get the grid index corresponding to the given player's type and the number of grid points"""
    idx = int(player_type * N_grid) + 1
    return idx


def meanVal(man_val, woman_val, value_fn, p, N_times):
    """
    Compute the expected continuation value for a man of type man_val
    in the first stage of the game who is matched with a woman of type woman_type.

    Inputs:
    - man_val: the type of a man
    - woman_val: the type of a woman
    - value_vn: a candidate for a value function,
    - p: the probability of rank-match
    - N_times: number of points on the grid

    Outputs: the comtinuation value.
    """

    # all possible types of players
    man_type = np.arange(N_times) / (N_times - 1)
    woman_type = np.arange(N_times) / (N_times - 1)

    # Defining matrices for integration
    m_ycontour = np.zeros((N_times, N_times))
    m_accept = np.zeros((N_times, N_times))

    for i in range(N_times):
        m_ycontour[:, i] = woman_type[i]

    m_reversed = np.zeros((N_times, N_times))
    i_val = type2index(man_val, N_times - 1)
    j_val = type2index(woman_val, N_times - 1)
    if i_val == N_times:
        m_reversed[(N_times - i_val + 1) :, j_val : (N_times + 1)] = 1
    else:
        m_reversed[: (N_times - i_val), :j_val] = 1
        m_reversed[(N_times - i_val + 1) :, j_val : (N_times + 1)] = 1

    m_direct = 1 - m_reversed

    for i in range(N_times):
        idx_accept = (value_fn <= man_type[i]) * (woman_type >= value_fn[i])

        m_accept[:, i] = np.flip(idx_accept)

    m_nonaccept = 1 - m_accept

    # Integrating over the types to find the expected value
    product_dir = np.multiply(
        m_accept * woman_val + np.multiply(m_nonaccept, m_ycontour), m_direct
    )
    product_rev = np.multiply(
        m_accept * woman_val + np.multiply(m_nonaccept, m_ycontour), m_reversed
    )

    # Expected value
    mean_val = np.sum(product_dir) / max(1, np.sum(m_direct)) * p + np.sum(
        product_rev
    ) / max(1, np.sum(m_reversed)) * (1 - p)

    return mean_val
