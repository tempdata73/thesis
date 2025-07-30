import math
import numpy as np

from src.utils import bezout_2d
from src.decorators import repeat_with_timeout


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
        x = rhs * x_b + t * gcd[idx]
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
                sol.append(x_last)
                return sol
            stack[-1][1] -= 1
            continue

        x_b_next, w_b_next = bezout_2d(q_first, gcd[idx + 1])
        lb_next = math.ceil(-rhs_next * x_b_next / gcd[idx + 1])
        ub_next = math.floor(rhs_next * w_b_next / q_first)

        # current parameter yielded a feasible interval
        if lb_next <= ub_next:
            sol.append(x)
            stack.append(
                [idx + 1, ub_next, lb_next, ub_next, rhs_next, x_b_next, w_b_next]
            )

        # no feasible interval given this parameter
        else:
            stack[-1][1] -= 1

    return None


@repeat_with_timeout()
def dioph(q, eta):
    while eta >= 0:
        x = solve_linear(q, eta)
        if x is not None:
            return np.asarray(x)

        eta -= 1

    return None


if __name__ == "__main__":
    from src.constants import bb_raw_options, bb_full_options
    from src.common import (
        stats_dioph_as_rhs_increases as dioph_rhs,
        stats_dp_as_rhs_increases as dp_rhs,
        stats_bb_as_rhs_increases as bb_rhs,
    )
    from src.common import run_parallel

    np.random.seed(42)

    p = np.random.randint(10, 10_000, size=1_000)
    g = math.gcd(*p)
    q = p // g
    m = g
    np.testing.assert_almost_equal(p, m * q)

    n = len(p)
    create_path = lambda name: f"times/fin/rhs/{name}/{n}n"  # noqa: E731

    rhs = np.round(p.sum() * np.linspace(0.01, 1.0, num=128)).astype(int)
    jobs = [
        {
            "name": "dioph",
            "log_path": create_path("dioph"),
            "job": lambda *_: dioph_rhs(dioph, q.copy(), m, rhs.copy()),
        },
        {
            "name": "dp",
            "log_path": create_path("dp"),
            "job": lambda *_: dp_rhs(p.copy(), rhs.copy()),
        },
        {
            "name": "bb_raw",
            "log_path": create_path("bb_raw"),
            "job": lambda *_: bb_rhs(p.copy(), rhs.copy(), **bb_raw_options.copy()),
        },
        {
            "name": "bb_full",
            "log_path": create_path("bb_full"),
            "job": lambda *_: bb_rhs(p.copy(), rhs.copy(), **bb_full_options.copy()),
        },
    ]
    run_parallel(jobs)
