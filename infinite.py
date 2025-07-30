import math
import numpy as np

from src.utils import bezout_2d
from src.decorators import repeat_with_timeout


def non_negative_int_sol(q, eta):
    n = len(q)
    x = np.zeros(n)

    for i in range(n - 2):
        g = math.gcd(*q[i + 1 :])
        x_b, w_b = bezout_2d(q[i], g)

        # feasible parameter and nonnegative solution
        t = math.ceil(-eta * x_b / g)
        x[i] = eta * x_b + g * t

        # update rhs and coprime vector
        eta = eta * w_b - q[i] * t
        q[i + 1 :] //= g

    x_bp, x_bl = bezout_2d(q[-2], q[-1])

    # feasible parameter
    b_1 = -eta * x_bp / q[-1]
    b_2 = eta * x_bl / q[-2]
    t = math.ceil(max(b_1, b_2))

    # last two nonnegative solutions
    x[-2] = eta * x_bp + q[-1] * t
    x[-1] = eta * x_bl - q[-2] * t

    return x


@repeat_with_timeout()
def dioph(q, eta):
    q = q.copy()
    x = np.zeros_like(q)

    # q_hat only has nonzero entries
    sigma = np.nonzero(q)
    q_hat = q[sigma]

    # q_hat satisfies q_hat[-2] < 0 < q_hat[-1]
    q_hat[0], q_hat[-1] = q_hat[-1], q_hat[0]
    j = np.where(q_hat[:-1] < 0)[0][0]
    q_hat[j], q_hat[-2] = q_hat[-2], q_hat[j]

    # x_hat is nonnegative and satisfies dot(q_hat, x_hat) == eta
    x_hat = non_negative_int_sol(q_hat, eta)

    # undo permutations
    x_hat[j], x_hat[-2] = x_hat[-2], x_hat[j]
    x_hat[0], x_hat[-1] = x_hat[-1], x_hat[0]

    # x is nonnegative and satisfies dot(q, x) == eta
    x[sigma] = x_hat

    return x


if __name__ == "__main__":
    from src.constants import bb_raw_options, bb_full_options
    from src.common import (
        stats_dioph_as_rhs_increases as dioph_rhs,
        stats_bb_as_rhs_increases as bb_rhs,
    )
    from src.common import run_parallel

    rhs = np.logspace(3, 7, num=128, base=10.0, dtype=int)

    p_int = np.array(
        [
            [9932, -9774, 9538, -9132],  # 3 decimal digits
            [99322, -97743, 95389, -91320],  # 4 decimal digits
            [993224, -977435, 953891, -913203],  # 5 decimal digits
        ]
    )

    # original projectively rational vector
    f = np.power(10.0, [-3, -4, -5])
    p = f[:, np.newaxis] * p_int

    # finding coprime multiples
    num_experiments = len(p_int)
    m = np.zeros(num_experiments)
    q = np.zeros_like(p_int)

    for i in range(num_experiments):
        g = math.gcd(*p_int[i])
        q[i] = p_int[i] // g
        m[i] = float(g) * f[i]

    # sanity check
    np.testing.assert_almost_equal(p, m[:, np.newaxis] * q)

    n = p.shape[1]
    create_path = lambda name, d: f"times/inf/{name}/{n}n-{d}d"  # noqa: E731
    jobs = []
    for i in range(num_experiments):
        jobs.extend(
            [
                {
                    "name": "dioph",
                    "log_path": create_path("dioph", i + 3),
                    "job": lambda *_: dioph_rhs(dioph, q[i].copy(), m[i], rhs.copy()),
                },
                {
                    "name": "bb_raw",
                    "log_path": create_path("bb_raw", i + 3),
                    "job": lambda *_: bb_rhs(
                        p[i].copy(), rhs.copy(), **bb_raw_options.copy()
                    ),
                },
                {
                    "name": "bb_full",
                    "log_path": create_path("bb_full", i + 3),
                    "job": lambda *_: bb_rhs(
                        p[i].copy(), rhs.copy(), **bb_full_options.copy()
                    ),
                },
            ]
        )
    run_parallel(jobs)
