import math
import numpy as np

from typing import Final


from src.utils import bezout_2d
from src.decorators import repeat_with_timeout
from src.common import stats_dioph_as_rhs_increases


NUM_REPS: Final[int] = 20
TIMEOUT: Final[int] = 5 * 60  # 5 minutes per instance


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


@repeat_with_timeout(NUM_REPS, TIMEOUT)
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
    # TODO: check dioph implementation is correct

    NUM_DECIMALS = 2

    # finding coprime multiple
    p_int = np.array([993, -977, 953, -913])
    g = math.gcd(*p_int)
    q = p_int // g
    m = g / math.pow(10, NUM_DECIMALS)

    # original projectively rational vector
    p = p_int / math.pow(10, NUM_DECIMALS)

    np.testing.assert_almost_equal(p, m * q)

    rhs = np.logspace(2, 7, num=128, base=10.0, dtype=int)
    stats = stats_dioph_as_rhs_increases(dioph, q, m, rhs)
