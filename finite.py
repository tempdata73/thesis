import math
import numpy as np
import matplotlib.pyplot as plt

import knapsack as kp
from utils import bezout_2d, repeat


# TODO: experiments:
# 1. time comparisons between mtu2, dioph, bb and dynamic model
#   1.1 as the dimension n increases
#   1.2 as the rhs increases ✔
# 2. actual number of layers visited vs feasibility zone ✔
# 4. comparison between search strategies (greedy vs centered)


def layer_bounds(q, m, u):
    q_star = np.max(q)
    eta = math.floor(u / m)
    tau = math.floor(math.floor(u / q_star) * q_star / m)
    return tau, eta


def solve_linear(q, rhs):
    n = len(q)

    # pre-compute gcds
    gcd = [1]
    for i in range(1, n - 1):
        g = math.gcd(*q[i:] // gcd[i - 1])
        gcd.append(g * gcd[i - 1])
    gcd.append(q[-1])  # dummy value

    # particular solutions
    x_b, w_b = bezout_2d(q[0], gcd[1])

    # bounds for feasible parameters
    lb = math.ceil(-rhs * x_b / g)
    ub = math.floor(rhs * w_b / q[0])

    # general solutions
    t = ub
    sol = []

    # pass-through stack
    idx = 0
    stack = [[idx, t, lb, ub, rhs, x_b, w_b]]

    while len(stack) > 0:
        idx, t, lb, ub, rhs, x_b, w_b = stack[-1]
        q_first = q[idx] // gcd[idx]

        x = rhs * x_b + t * g
        rhs_next = rhs - q[idx] * x

        # current parameter is outside feasible bound. backtracking
        if t < lb:
            stack.pop()
            if len(stack) == 0:
                break
            sol.pop()
            stack[-1][1] -= 1
            continue

        # on to the last solution
        if idx == n - 1:
            x_last, res = divmod(rhs, q[idx])
            if res == 0:
                return [*sol, x_last]
            stack[-1][1] -= 1
            continue

        x_b_next, w_b_next = bezout_2d(q_first, gcd[idx + 1])
        lb_next = math.ceil(-rhs_next * x_b_next / gcd[idx + 1])
        ub_next = math.floor(rhs_next * w_b_next / q_first)

        # current parameter yielded a feasible interval
        if lb_next <= ub_next:
            sol.append(x)
            stack.append([
                idx + 1,
                ub_next,
                lb_next,
                ub_next,
                rhs_next,
                x_b_next,
                w_b_next
            ])

        # no feasible interval given this parameter
        else:
            stack[-1][1] -= 1

    return None


@repeat(num_iter=20)
def solve_dioph(q, eta):
    while eta >= 0:
        x  = solve_linear(q, eta)
        if x is not None:
            return np.asarray(x)

        eta -= 1

    return None


def experiment_1(q, m):
    """
    number of layers visited vs feasibility zone
    """

    num_samples = 128
    prop = np.sum(q) * np.logspace(start=-5, stop=-4, base=10, num=num_samples)
    visited = {
        "pctg": np.zeros(num_samples),
        "abs": np.zeros(num_samples),
    }
    to_visit = np.zeros(num_samples)

    for i, u in enumerate(prop):
        tau, eta = layer_bounds(q, m, u)
        k, x = solve_dioph(q, eta)
        np.testing.assert_allclose(np.dot(q, x), k)
        num_visited = eta - np.dot(q, x)

        visited["abs"][i] = num_visited
        to_visit[i] = eta - tau
        if eta != tau:
            visited["pctg"][i] = num_visited / (eta - tau)

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
    ax[0].plot(prop, visited["pctg"])
    ax[1].plot(prop, visited["abs"], label="did")
    ax[1].plot(prop, to_visit, label="must")
    ax[1].legend()
    plt.show()


def experiment_2(size=100):
    """
    termination times as rhs increases
    """
    n = 128
    np.random.seed(42)

    stats = {
        "capacity": [0] * n,
        "bb_raw": {
            "mu": [0] * n,
            "sigma": [0] * n,
            "obj": [0] * n,
        },
        "bb_full": {
            "mu": [0] * n,
            "sigma": [0] * n,
            "obj": [0] * n,
        },
        "dp": {
            "mu": [0] * n,
            "sigma": [0] * n,
            "obj": [0] * n,
        },
        "dioph": {
            "mu": [0] * n,
            "sigma": [0] * n,
            "obj": [0] * n,
        },
    }

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

    p = np.sort(np.random.randint(10, 10 * size, size=size))
    stats["price"] = p.copy()
    m = math.gcd(*p)
    q = (p // m).astype(int)

    np.testing.assert_allclose(p, m * q)

    print(f"experiment_2 with dimension = {size}")
    rhs = np.linspace(0.0, 1.0, 128) * q.sum()
    for i, capacity in enumerate(rhs):
        capacity = int(capacity)
        print(f"on problem with capacity = {capacity} ({i + 1}/{n})")
        stats["capacity"][i] = capacity

        # diophantine method
        _, eta = layer_bounds(q, m, capacity)
        x_dioph, (mu, sigma) = solve_dioph(q, eta)
        obj_dioph = np.dot(x_dioph, p)
        stats["dioph"]["mu"][i] = mu
        stats["dioph"]["sigma"][i] = sigma
        stats["dioph"]["obj"][i] = obj_dioph

        # dynamic method
        x_dp, (mu, sigma) = kp.ukp_dp(p, capacity)
        obj_dp = np.dot(x_dp, p)
        stats["dp"]["mu"][i] = mu
        stats["dp"]["sigma"][i] = sigma
        stats["dp"]["obj"][i] = obj_dp

        # bb method (cuts on)
        x_bb_full, (mu, sigma) = kp.ukp_bb(p, capacity, **bb_full_options)
        obj_bb_full = np.dot(x_bb_full, p)
        stats["bb_full"]["mu"][i] = mu
        stats["bb_full"]["sigma"][i] = sigma
        stats["bb_full"]["obj"][i] = obj_bb_full

        # bb method (cuts off)
        x_bb_raw, (mu, sigma) = kp.ukp_bb(p, capacity, **bb_raw_options)
        obj_bb_raw = np.dot(x_bb_raw, p)
        stats["bb_raw"]["mu"][i] = mu
        stats["bb_raw"]["sigma"][i] = sigma
        stats["bb_raw"]["obj"][i] = obj_bb_raw

        if not np.isclose(obj_dioph, obj_bb_full):
            print("[ERROR]: different objective values")
            print(f"[DEBUG]: {obj_dioph=}")
            print(f"[DEBUG]: {obj_dp=}")
            print(f"[DEBUG]: {obj_bb_full=}")
            print(f"[DEBUG]: {obj_bb_raw=}")

    np.save(f"times/poscase-dim-{size}", stats)

if __name__ == "__main__":
    experiment_2(size=100)
