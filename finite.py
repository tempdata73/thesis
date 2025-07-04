import math
import numpy as np
import matplotlib.pyplot as plt

from utils import bezout_2d, repeat


# TODO: experiments:
# 1. time comparisons between mtu2, dioph, bb and dynamic model
#   1.1 as the dimension n increases
#   1.2 as the rhs increases
# 2. actual number of layers visited vs feasibility zone
# 3. time vs number of decimal digits
# 4. comparison between search strategies (greedy vs centered)


def layer_bounds(q, m, u):
    q_star = np.max(q)
    eta = math.floor(u / m)
    tau = math.floor(math.floor(u / q_star) * q_star / m)
    return tau, eta


def solve_linear_eq(q, rhs):
    n = len(q)

    # particular solutions
    g = math.gcd(*q[1:])
    x_b, w_b = bezout_2d(q[0], g)

    # bounds for feasible parameters
    lb = math.ceil(-rhs * x_b / g)
    ub = math.floor(rhs * w_b / q[0])

    # general solutions
    t = ub
    sol = []

    # pass-through stack
    idx = 0
    stack = [[idx, t, lb, ub, rhs, g, x_b, w_b]]

    while len(stack) > 0:
        idx, t, lb, ub, rhs, g, x_b, w_b = stack[-1]
        q_rest = q[idx:] // g
        rhs_next = rhs * w_b - t * q_rest[0]

        # current parameter is outside feasible bound. backtracking
        if t > ub:
            stack.pop()
            if len(stack) == 0:
                break
            sol.pop()
            stack[-1][1] += 1
            continue

        # on to the last solution
        if idx == n - 1:
            x_last, res = divmod(rhs, q[idx])
            if res == 0:
                return [*sol, x_last]
            stack[-1][1] += 1
            continue

        g_next = math.gcd(*q_rest[1:])
        x_b_next, w_b_next = bezout_2d(q_rest[0], g_next)

        lb_next = math.ceil(-rhs_next * x_b_next / g_next)
        ub_next = math.floor(rhs_next * w_b_next / q_rest[0])

        # current parameter yielded a feasible interval
        if lb_next <= ub_next:
            sol.append(rhs * x_b + t * g)
            stack.append([
                idx + 1,
                ub_next,
                lb_next,
                ub_next,
                rhs_next,
                g * g_next,
                x_b_next,
                w_b_next
            ])

        # no feasible interval given this parameter
        else:
            stack[-1][1] += 1

    return None


@repeat(num_iter=30)
def solve_dioph(q, eta):
    while eta >= 0:
        x = solve_linear_eq(q, eta)

        if x is not None:
            return eta, np.asarray(x)

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


if __name__ == "__main__":
    # from knapsack import ukp_dp

    np.random.seed(42)

    p = np.sort(np.random.randint(1, 1000, size=100_000))
    m = math.gcd(*p)
    q = (p // m).astype(int)
    np.testing.assert_allclose(p, m * q)
    # np.testing.assert_array_less(np.zeros_like(q), q)

    u = math.floor(0.01 * sum(q))
    # ukp_dp(p, u)
    _, eta = layer_bounds(q, m, u)
    solve_dioph(q, eta)
    # experiment_1(q, m)
