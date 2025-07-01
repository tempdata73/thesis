import math
import numpy as np
import matplotlib.pyplot as plt

from utils import bezout_2d


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
    if len(q) == 2:
        x_prime, y_prime = bezout_2d(q[0], q[1])
        lb = math.ceil(-rhs * x_prime / q[1])
        ub = math.floor(rhs * y_prime / q[0])

        if ub < lb:
            return None

        x = rhs * x_prime + lb * q[1]
        y = rhs * y_prime - lb * q[0]
        return (x.item(), y.item())

    g = math.gcd(*q[1:])
    x_prime, omega_prime = bezout_2d(q[0], g)

    # bounds for feasible parameters
    lb = math.ceil(-rhs * x_prime / g)
    ub = math.floor(rhs * omega_prime / q[0])

    for t in range(lb, ub + 1):
        omega = rhs * omega_prime - t * q[0]
        x = rhs * x_prime + t * g
        rest = solve_linear_eq((q[1:] / g).astype(int), omega)

        if rest is not None:
            return (x, *rest)
    return None


def solve_dioph(q, eta):
    while eta >= 0:
        x = solve_linear_eq(q, eta)
        eta -= 1

        if x is not None:
            return np.asarray(x)

    print(f"problem is not feasible with rhs = {eta}")
    return None


def experiment_1(q, m):
    """
    number of layers visited vs feasibility zone
    """

    num_samples = 2048
    prop = np.sum(q) * np.logspace(start=-3, stop=-2, base=10, num=num_samples)
    visited = {
        "pctg": np.zeros(num_samples),
        "abs": np.zeros(num_samples),
    }
    to_visit = np.zeros(num_samples)

    for i, u in enumerate(prop):
        tau, eta = layer_bounds(q, m, u)
        x = solve_dioph(q, eta)
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
    np.random.seed(42)

    p = np.random.randint(1, 10_000, size=50)
    m = math.gcd(*p)
    q = (p // m).astype(int)
    np.testing.assert_allclose(p, m * q)
    # np.testing.assert_array_less(np.zeros_like(q), q)

    experiment_1(q, m)
