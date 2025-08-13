import numpy as np
# from .decorators import repeat_with_timeout


# @repeat_with_timeout()
def ukp_dp(prices: list[int], capacity: int):
    """
    dynamic programming model for the unbounded knapsack problem.
    taken from: martello and toth: knapsack problems.

    adapted to solve the unbounded set sum problem and to retrieve
    the solution.
    """

    prices = np.sort(prices)
    cap = np.arange(capacity + 1)

    # initialize
    take = cap // prices[0]
    obj = take * prices[0]
    choice = np.where(take > 0, 0, -1)
    print(take)
    print(choice)

    # update
    for j in range(1, len(prices)):
        p_j = prices[j]

        for c in cap[p_j:]:
            take = obj[c - p_j] + p_j
            print(choice)

            if take > obj[c]:
                obj[c] = take
                choice[c] = j

    # reconstruct solution
    counts = np.zeros_like(prices)
    print(choice)
    while (0 < capacity) and (choice[capacity] != -1):
        i = choice[capacity]
        counts[i] += 1
        capacity -= prices[i]

    return counts
