import math


def ukp_dp(prices: list[int], capacity: int):
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

    optimal = obj.pop()
    return counts, optimal


def upper_bound(profits: list[int], weights: list[int], capacity: int) -> int:
    """
    martello & toth: p. 92
    """

    if capacity <= 0:
        return 0
    elif len(weights) < 3:
        return (capacity // weights[0]) * profits[0]

    w_1, w_2, w_3 = weights[:3]
    p_1, p_2, p_3 = profits[:3]

    c_1 = capacity
    c_2 = c_1 % w_1  # (3.18)
    c_3 = c_2 % w_2  # (3.22)

    z = (c_1 // w_1) * p_1 + (c_2 // w_2) * p_2  # (3.21)

    k = math.ceil((w_2 - c_3) / w_1)
    u_0 = z + (c_3 * p_3) // w_3  # (3.23)
    u_hat = z + math.floor((c_3 + k * w_1) * (p_2 / w_2) - k * p_1)  # (3.25)

    return max(u_0, u_hat)  # (3.26)


def mtu2(
    weights: list[int],
    profits: list[int],
    capacity: int,
) -> tuple[int, list[int]]:

    n = len(weights)
    best_val = 0
    best_cnt = [0] * n
    cur_cnt = [0] * n

    # TODO: since p[i] = w[i] we can optimize this in the modified version
    order = sorted(range(n), key=lambda i: profits[i] / weights[i], reverse=True)
    w = [weights[i] for i in order]
    p = [profits[i] for i in order]

    def dfs(i: int, cap: int, val: int) -> None:
        nonlocal best_val, best_cnt

        if i == n or cap == 0:
            if val > best_val:
                best_val = val
                best_cnt = cur_cnt.copy()
            return

        # ub = val + _upper_bound_U3(cap, w[i:], p[i:])
        ub = val + upper_bound(p[i:], w[i:], cap)
        if ub <= best_val:
            return

        max_k = cap // w[i]
        for k in range(max_k, -1, -1):
            if k:
                cur_cnt[i] = k
            else:
                cur_cnt[i] = 0
            dfs(
                i + 1,
                cap - k * w[i],
                val + k * p[i],
            )

    dfs(0, capacity, 0)

    orig_choice = [0] * len(weights)
    for o_idx, m_idx in enumerate(order):
        orig_choice[m_idx] = best_cnt[o_idx]

    return best_val, orig_choice


if __name__ == "__main__":
    weights = [4, 3, 6]
    profits = [6, 5, 10]
    capacity = 10
    opt_val, opt_cnt = mtu2(weights, profits, capacity)
    print("optimal profit =", opt_val)
    print("item counts   =", opt_cnt)

    print(upper_bound([20, 5, 1], [10, 5, 3], 39))
