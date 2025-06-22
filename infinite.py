import math
import numpy as np
import pulp as lp

from utils import bezout_2d, repeat


NUM_ITER = 10


def feasible_point(c_p, eta, flag):
    n = len(c_p)
    x = np.zeros(n)

    omega = eta
    for i in range(n - 2):
        g = math.gcd(*c_p[1:])
        x_bezout, omega_bezout = bezout_2d(c_p[i], g)

        # get feasible parameter
        if (i == n - 3) and ((flag == 1) or (flag == 4)):
            c_1 = omega * omega_bezout / c_p[i]
            c_2 = 2.0 * c_p[-2] * c_p[-1] / (c_p[i] * g * g)
            if flag == 1:
                t = math.ceil(max(-omega * x_bezout / g, c_1 + c_2))
            elif flag == 4:
                t = math.ceil(max(-omega * x_bezout / g, c_1 - c_2))
        else:
            t = math.ceil(-omega * x_bezout / g)

        # update
        x[i] = omega * x_bezout + g * t
        omega = omega * omega_bezout - c_p[i] * t
        c_p[i:] //= g

    # last two solutions
    x_prime, y_prime = bezout_2d(c_p[-2], c_p[-1])
    b_1 = -omega * x_prime / c_p[-1]
    b_2 = -omega * y_prime / c_p[-2]
    match flag:
        case 1:
            t = math.floor(b_1)
        case 2:
            t = math.ceil(max(b_1, b_2))
        case 3:
            t = math.floor(min(b_1, b_2))
        case 4:
            t = math.ceil(b_1)
    x[-2] = omega * x_prime + c_p[-1] * t
    x[-1] = omega * y_prime - c_p[-2] * t

    return x


@repeat(num_iter=NUM_ITER)
def solve_dioph(c_p, k, s):
    c_p = c_p.copy()
    eta = math.floor(s / k)

    # strategy for finding feasible point is divided in four cases
    # depending on the sign of last two entries of c_p
    obj_second_to_last = c_p[-2]
    obj_last = c_p[-1]
    if (obj_second_to_last < 0.0) and (obj_last < 0.0):
        flag = 1
    elif (obj_second_to_last < 0.0) and (obj_last > 0.0):
        flag = 2
    elif (obj_second_to_last > 0.0) and (obj_last < 0.0):
        flag = 3
    elif (obj_second_to_last > 0.0) and (obj_last > 0.0):
        flag = 4

    # permuting entries of c in order to find feasible t_{n-2}
    switch = None
    if flag == 1:
        c_p[-3], c_p[0] = c_p[0], c_p[-3]
        switch = 0  # first entry of c_p is always positive
    elif flag == 4:
        neg_indices = np.where(c_p[:-2] < 0.0)[0]
        assert len(neg_indices) > 0
        switch = neg_indices[0].item()
        c_p[-3], c_p[switch] = c_p[switch], c_p[-3]

    x = feasible_point(c_p, eta, flag=flag)
    # invert initial permutation
    if (flag == 1) or (flag == 4):
        x[switch], x[-3] = x[-3], x[switch]

    return x


@repeat(num_iter=NUM_ITER)
def solve_pulp(p, s, options=None):
    # create lp problem
    x = [lp.LpVariable(f"b_{i}", lowBound=0, cat=lp.LpInteger) for i in range(len(p))]
    prob = lp.LpProblem(sense=lp.LpMaximize)
    prob += lp.lpDot(p, x), "objective"
    prob += lp.lpDot(p, x) <= s, "constraint"

    # solve problem
    cuts = options is None
    prob.solve(lp.PULP_CBC_CMD(msg=False, cuts=cuts, options=options))

    res = np.array([lp.value(var) for var in x])
    return res


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    NUM_DECIMALS = 2
    FILENAME = f"neg-4d-{NUM_DECIMALS}dig"
    PULP_OPTIONS = [
        "gomory on",
        "knapsack off",
        "probing off",
    ]

    p = np.array([9.93, 9.77, -9.53, -9.13])

    # finding coprime multiple of projectively rational vector
    p_1 = np.array([993, 977, -953, -913])
    g = math.gcd(*p_1)
    c_p = p_1 // g
    multiplier = g / math.pow(10, NUM_DECIMALS)
    np.testing.assert_almost_equal(p, multiplier * c_p)

    slack = np.logspace(2, 4, num=100, base=10.0, dtype=int)
    stats: dict[str, dict[str, list[float]]] = {
        "pulp": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
        "pulp-cuts": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
        "dioph": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
    }

    for i, s in enumerate(slack):
        print(f"[INFO]: on problem {i + 1} with slack {s}")

        # pure pulp method
        print("[INFO]: solving with pulp")
        pulp_sol, (mu, std) = solve_pulp(p, s)
        stats["pulp"]["mean"][i] = mu
        stats["pulp"]["std"][i] = std

        # pulp method with gomory cuts
        print("[INFO]: solving with pulp + cuts")
        _, (mu, std) = solve_pulp(p, s, options=PULP_OPTIONS)
        stats["pulp-cuts"]["mean"][i] = mu
        stats["pulp-cuts"]["std"][i] = std

        # diophantine method
        print("[INFO]: solving with diophantine")
        dioph_sol, (mu, std) = solve_dioph(c_p, multiplier, s)
        stats["dioph"]["mean"][i] = mu
        stats["dioph"]["std"][i] = std
        assert np.all(dioph_sol >= 0.0), "non-negativity is violated"

        # cannot guarantee both solutions to be equal
        dioph_obj = np.dot(p, dioph_sol)
        pulp_obj = np.dot(p, pulp_sol)

        if not np.isclose(dioph_obj, pulp_obj):
            print(f"[ERROR]: objective values don't match for slack {s}")
            print(f"[DEBUG]: {pulp_sol=} -> obj = {pulp_obj}")
            print(f"[DEBUG]: {dioph_sol=} -> obj = {dioph_obj}")
            print(f"[DEBUG]: dioph_obj > pulp_obj is {dioph_obj > pulp_obj}")
        print()

    # plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(slack, stats["pulp"]["mean"], "o--", c="firebrick", zorder=5, label="b&b")
    ax.loglog(
        slack, stats["pulp-cuts"]["mean"], "o--", c="indianred", zorder=5, label="gom"
    )
    ax.loglog(
        slack, stats["dioph"]["mean"], "o--", c="navy", zorder=5, label="diophantine"
    )
    ax.grid()

    ax.set_ylabel("time [ns]")
    ax.set_xlabel("slack")
    ax.legend()

    plt.savefig(f"figs/{FILENAME}.pdf")
    with open(f"times/{FILENAME}.json", "w") as outfile:
        json.dump(stats, outfile)
