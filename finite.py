import math
import numpy as np

from numba import njit
from src.utils import bezout_2d, cumgcd
from src.decorators import repeat_with_timeout


@njit
def non_neg_int_sol(q, rhs, gcd, bez):
    n = len(q)

    # particular solutions
    x_b, w_b = bez[0]

    # feasible parameters
    lb = math.ceil(-rhs * x_b / gcd[1])
    ub = math.floor(rhs * w_b / q[0])

    # pass-through stack
    sol = np.empty(n, dtype=np.int64)
    stack = [(0, lb, ub, rhs)]

    while len(stack) > 0:
        i, lb, ub, rhs = stack.pop()

        # search space is empty. backtracking
        if ub < lb:
            continue

        # binary search on same level
        t = (ub + lb) // 2
        if t + 1 <= ub:
            stack.append((i, t + 1, ub, rhs))
        if t - 1 >= lb:
            stack.append((i, lb, t - 1, rhs))

        # prev equation solutions
        x_b, w_b = bez[i]
        x_p = rhs * x_b + t * gcd[i + 1]
        rhs = rhs * w_b - t * q[i]

        # last equation
        if i == n - 3:
            x_b, y_b = bez[-1]

            # check if there exists a feasible parameter
            lb = math.ceil(-rhs * x_b / q[-1])
            ub = math.floor(rhs * y_b / q[-2])

            if ub < lb:
                continue

            # last two nonnegative solutions
            x = rhs * x_b + lb * q[-1]
            y = rhs * y_b - lb * q[-2]

            # unpack previous solutions
            sol[-3:] = x_p, x, y
            return sol

        # feasible parameters
        x_b, w_b = bez[i + 1]
        lb = math.ceil(-rhs * x_b / gcd[i + 2])
        ub = math.floor(rhs * w_b / q[i + 1])

        # push next level
        if lb <= ub:
            sol[i] = x_p
            stack.append((i + 1, lb, ub, rhs))

    return None


@njit
def preprocess(q):
    q = q.copy()
    gcd = [1]
    bez = []

    for i in range(len(q) - 2):
        g = cumgcd(q[i + 1 :])
        q[i + 1 :] //= g
        gcd.append(g * gcd[i])
        bez.append(bezout_2d(q[i], g))
    bez.append(bezout_2d(q[-2], q[-1]))

    return gcd, bez


@repeat_with_timeout()
def dioph(q, eta):
    # preprocessing: particular solution to dot(q, x) == 1
    gcd, bez = preprocess(q)

    # find first nonnegative solution to dot(q, x) = eta
    while eta >= 0:
        x = non_neg_int_sol(q, eta, gcd, bez)
        if x is not None:
            return x
        eta -= 1

    return None


if __name__ == "__main__":
    from functools import partial
    from src.common import (
        stats_dioph_as_rhs_increases as dioph_rhs,
    )
    from src.common import run_parallel
    from src.constants import RANDOM_SEED

    rng = np.random.default_rng(seed=RANDOM_SEED)
    p = rng.integers(10, 10_000, size=1_000)
    g = math.gcd(*p)
    q = p // g
    m = np.int64(g)
    np.testing.assert_almost_equal(p, m * q)

    n = len(p)
    create_rhs_path = lambda name: f"times/fin/rhs/{name}/{n}n"  # noqa: E731
    create_dim_path = lambda name: f"times/fin/dim/{name}/mt"  # noqa: E731

    rhs = np.round(p.sum() * np.linspace(0.01, 1.0, num=128)).astype(int)
    dims = np.array(
        [
            50,
            100,
            200,
            500,
            1_000,
            2_000,
            5_000,
            10_000,
            20_000,
            30_000,
            40_000,
            50_000,
            60_000,
            70_000,
            80_000,
            90_000,
            100_000,
            150_000,
            200_000,
            250_000,
        ]
    )
    jobs = [
        {
            "name": "dioph",
            "log_path": create_rhs_path("dioph"),
            "job": partial(dioph_rhs, dioph, q.copy(), m, rhs.copy()),
        },
    ]
    run_parallel(jobs)
