import numpy as np
import scipy.linalg as la

from src.utils import bezout_2d, cumgcd


def lattice_projection(q):
    n: int = len(q)

    prod: int = 1
    g = np.zeros(n - 1, dtype=int)
    for i in range(n - 1):
        g[i] = cumgcd(q[i:] // prod)
        prod *= g[i]
    assert g[0] == 1, "q-vector is not coprime"

    # bezout coefficients
    x_bez = np.zeros(n, dtype=int)
    w_bez = np.zeros(n - 2, dtype=int)
    prod: int = 1
    for i in range(n - 2):
        x_bez[i], w_bez[i] = bezout_2d(q[i] // prod, g[i + 1])
        prod *= g[i + 1]

    # \nu vector
    nu = np.zeros(n, dtype=int)
    nu[0] = x_bez[0]
    nu[1] = x_bez[1]

    prod = np.cumprod(w_bez)
    for i in range(2, n):
        nu[i] = x_bez[i] * prod[i - 2]

    print(f"{nu = }")
    assert np.dot(q, nu) == 1, "nu-vector is not a particular solution"

    return nu


if __name__ == "__main__":
    q = np.array([2, 3, 8, 5])
    print(lattice_projection(q.copy()))
