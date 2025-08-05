import math
import numpy as np

from src.utils import bezout_2d
from src.decorators import repeat_with_timeout


def non_negative_int_sol(q, rhs, gcd, bez):
    n = len(q)

    # particular solutions
    x_b, w_b = bez[0]

    # feasible parameters
    lb = math.ceil(-rhs * x_b / gcd[1])
    ub = math.floor(rhs * w_b / q[0])

    # pass-through stack
    stack = [[0, ub, lb, rhs, None]]

    while len(stack) > 0:
        i, t, lb, rhs, x = stack[-1]

        # search space is empty. backtracking
        if t < lb:
            stack.pop()
            continue

        # update stack for next feasible parameter
        stack[-1][1] -= 1

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
            sol = [stack[j][-1] for j in range(1, i + 1)] + [x_p, x, y]
            return sol

        # feasible parameters
        x_b, w_b = bez[i + 1]
        lb = math.ceil(-rhs * x_b / gcd[i + 2])
        ub = math.floor(rhs * w_b / q[i + 1])

        # update stack
        stack.append([i + 1, ub, lb, rhs, x_p])

    return None


@repeat_with_timeout()
def dioph(q, eta):
    _q = q.copy()

    # preprocessing: particular solutions to dot(q, x) == 1
    gcd = [1]
    bez = []
    for i in range(len(_q) - 2):
        g = math.gcd(*_q[i + 1:])
        _q[i + 1:] //= g

        gcd.append(g * gcd[i])
        bez.append(bezout_2d(_q[i], g))

    bez.append(bezout_2d(_q[-2], _q[-1]))

    # find first nonnegative solution
    while eta >= 0:
        x = non_negative_int_sol(_q, eta, gcd, bez)
        if x is not None:
            sol = np.asarray(x)
            if np.any(sol < 0):
                print("nonnegativity not enforced")

            if np.dot(q, sol) != eta:
                print("equation not enforced")
                print(np.dot(q, sol), "\\neq", eta)

            return sol
        eta -= 1

    return None


if __name__ == "__main__":
    n = (5, 10, 100, 1_000, 5_000, 10_000, 25_000, 50_000)
    np.random.seed(42)
    for dim in n:
        print(f"on dimension {dim}")

        p = np.sort(np.random.randint(10, 10 * dim, size=dim))[::-1]
        m = math.gcd(*p)
        q = p // m
        np.testing.assert_equal(p, q * m)

        eta = int(0.1 * q.sum())
        x, (mu, sigma), timeout = dioph(q, eta)
        print(f"took {mu * 1e-6:0.4f} ± {sigma * 1e-6:0.4f} ms to find solution")
        print(f"timed out = {timeout}")


    # from src.constants import bb_raw_options, bb_full_options
    # from src.common import (
    #     stats_dioph_as_rhs_increases as dioph_rhs,
    #     stats_dp_as_rhs_increases as dp_rhs,
    #     stats_bb_as_rhs_increases as bb_rhs,
    # )
    # from src.common import run_parallel

    # np.random.seed(42)

    # p = np.random.randint(10, 10_000, size=1_000)
    # g = math.gcd(*p)
    # q = p // g
    # m = np.int64(g)
    # np.testing.assert_almost_equal(p, m * q)

    # n = len(p)
    # create_path = lambda name: f"times/fin/rhs/{name}/{n}n"  # noqa: E731

    # rhs = np.round(p.sum() * np.linspace(0.01, 1.0, num=128)).astype(int)
    # jobs = [
    #     {
    #         "name": "dioph",
    #         "log_path": create_path("dioph"),
    #         "job": lambda *_: dioph_rhs(dioph, q.copy(), m, rhs.copy()),
    #     },
    #     {
    #         "name": "dp",
    #         "log_path": create_path("dp"),
    #         "job": lambda *_: dp_rhs(p.copy(), rhs.copy()),
    #     },
    #     {
    #         "name": "bb_raw",
    #         "log_path": create_path("bb_raw"),
    #         "job": lambda *_: bb_rhs(p.copy(), rhs.copy(), **bb_raw_options.copy()),
    #     },
    #     {
    #         "name": "bb_full",
    #         "log_path": create_path("bb_full"),
    #         "job": lambda *_: bb_rhs(p.copy(), rhs.copy(), **bb_full_options.copy()),
    #     },
    # ]
    # run_parallel(jobs)
