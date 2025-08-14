import os
import math
import logging

from numba import njit


def setup_logger(log_path):
    os.makedirs(log_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    filename = os.path.join(log_path, "logs")
    handler = logging.FileHandler(filename, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@njit
def bezout_2d(a: int, b: int) -> tuple[int, int]:
    sgn_a = int(math.copysign(1.0, a))
    sgn_b = int(math.copysign(1.0, b))
    a, b = abs(a), abs(b)

    prev_r, r = a, b
    prev_x, x = 1, 0
    prev_y, y = 0, 1

    while r != 0:
        q = prev_r // r

        prev_r, r = r, prev_r - q * r
        prev_x, x = x, prev_x - q * x
        prev_y, y = y, prev_y - q * y

    return sgn_a * prev_x, sgn_b * prev_y


@njit
def cumgcd(q):
    g = q[0]
    for i in range(1, len(q)):
        g = math.gcd(g, q[i])
    return g
