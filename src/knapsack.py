import numpy as np
from .decorators import repeat_with_timeout
from numba import njit


@repeat_with_timeout()
@njit
def ukp_dp(prices, capacity):
    indices = np.argsort(prices)
    prices = prices[indices].copy()

    cap = np.arange(capacity + 1)

    # initialize
    take = cap // prices[0]
    obj = take * prices[0]
    choice = np.where(take > 0, 0, -1)

    # update
    for j in range(1, len(prices)):
        p_j = prices[j]

        for c in cap[p_j:]:
            take = obj[c - p_j] + p_j

            if take > obj[c]:
                obj[c] = take
                choice[c] = j

    # reconstruct solution
    counts = np.zeros_like(prices)
    while (0 < capacity) and (choice[capacity] != -1):
        i = choice[capacity]
        counts[i] += 1
        capacity -= prices[i]

    return counts[np.argsort(indices)]
