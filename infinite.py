import math
import numpy as np
import pulp as lp

from utils import bezout_2d, repeat


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


@repeat(num_iter=20)
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


@repeat(num_iter=20)
def solve_pulp(p, s, **kwargs):
    # create lp problem
    x = [lp.LpVariable(f"b_{i}", lowBound=0, cat=lp.LpInteger) for i in range(len(p))]
    prob = lp.LpProblem(sense=lp.LpMaximize)
    prob += lp.lpDot(p, x), "objective"
    prob += lp.lpDot(p, x) <= s, "constraint"

    # solve problem
    prob.solve(lp.PULP_CBC_CMD(msg=False, **kwargs))

    res = np.array([lp.value(var) for var in x])
    return res


if __name__ == "__main__":
    import json

    NUM_DECIMALS = 2
    FILENAME = f"neg-4d-{NUM_DECIMALS}dig"
    bb_raw_options = {
        "presolve": True,
        "cuts": False,
        "options": None,
    }

    bb_full_options = {
        "presolve": True,
        "cuts": True,
        "options": ["gomory on", "knapsack off", "probing off"],
    }

    p = np.array([9.93, -9.77, 9.53, -9.13])

    # finding coprime multiple of projectively rational vector
    p_1 = np.array([993, -977, 953, -913])
    g = math.gcd(*p_1)
    c_p = p_1 // g
    multiplier = g / math.pow(10, NUM_DECIMALS)
    np.testing.assert_almost_equal(p, multiplier * c_p)

    n = 128
    slack = np.logspace(2, 3, num=n, base=10.0, dtype=int)
    stats: dict[str, dict[str, list[float]]] = {
        "bb_full": {
            "mu": [0.0] * n,
            "sigma": [0.0] * n,
            "obj": [0.0] * n,
        },
        "bb_raw": {
            "mu": [0.0] * n,
            "sigma": [0.0] * n,
            "obj": [0.0] * n,
        },
        "dioph": {
            "mu": [0.0] * n,
            "sigma": [0.0] * n,
            "obj": [0.0] * n,
        },
    }

    for i, s in enumerate(slack):
        print(f"[INFO]: on slack {s} ({i + 1}/{n})")

        # full options
        x_bb_full, (mu, sigma) = solve_pulp(p, s, **bb_full_options)
        obj_bb_full = np.dot(p, x_bb_full)
        stats["bb_full"]["mu"][i] = mu
        stats["bb_full"]["sigma"][i] = sigma
        stats["bb_full"]["obj"][i] = obj_bb_full

        # pure implementation
        x_bb_raw, (mu, sigma) = solve_pulp(p, s, **bb_raw_options)
        obj_bb_raw = np.dot(p, x_bb_raw)
        stats["bb_raw"]["mu"][i] = mu
        stats["bb_raw"]["sigma"][i] = sigma
        stats["bb_raw"]["obj"][i] = obj_bb_raw

        # diophantine method
        x_dioph, (mu, sigma) = solve_dioph(c_p, multiplier, s)
        obj_dioph = np.dot(p, x_dioph)
        stats["dioph"]["mu"][i] = mu
        stats["dioph"]["sigma"][i] = sigma
        stats["dioph"]["obj"][i] = obj_dioph

        if not np.isclose(obj_dioph, obj_bb_full):
            print(f"[ERROR]: objective values don't match for slack {s}")
            print(f"[DEBUG]: {x_bb_full=} -> obj = {obj_bb_full}")
            print(f"[DEBUG]: {x_dioph=} -> obj = {obj_dioph}")

    with open(f"times/inf/{FILENAME}.json", "w") as outfile:
        json.dump(stats, outfile)
