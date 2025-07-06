import pulp as pl
import numpy as np

from utils import repeat


@repeat(num_iter=20)
def ukp_dp(prices: list[int], capacity: int) -> list[int]:
    """
    dynamic programming model for the unbounded knapsack problem.
    taken from: martello and toth: knapsack problems.

    adapted to solve the unbounded set sum problem and to retrieve
    the solution.
    """

    # initialize
    p_1 = prices[0]
    obj = [0] * (capacity + 1)
    choice = [-1] * (capacity + 1)

    for cap in range(capacity + 1):
        k = cap // p_1
        obj[cap] = k * p_1
        if k > 0:
            choice[cap] = 0

    # update
    for m in range(1, len(prices)):
        p_m = prices[m]

        for cap in range(p_m, capacity):
            take = obj[cap - p_m] + p_m

            if take > obj[cap]:
                obj[cap] = take
                choice[cap] = m

    # reconstruct solution
    counts = [0] * len(prices)
    cap = capacity
    while (0 < cap) and (choice[cap] != -1):
        i = choice[cap]
        counts[i] += 1
        cap -= prices[i]

    return counts


@repeat(num_iter=20)
def ukp_bb(
    prices: list[int], capacity: int, **kwargs
):
    n = len(prices)
    prob = pl.LpProblem(sense=pl.LpMaximize)
    x = [pl.LpVariable(name=f"x_{i}", lowBound=0, cat=pl.LpInteger) for i in range(n)]
    prob += pl.lpDot(prices, x), "obj"
    prob += pl.lpDot(prices, x) <= capacity, "c"
    solver = pl.PULP_CBC_CMD(msg=False, **kwargs)
    prob.solve(solver)
    return np.fromiter((var.varValue for var in x), dtype=int)
