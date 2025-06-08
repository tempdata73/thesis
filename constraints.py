import math
import numpy as np
import scipy.linalg as la


def _bezout(a: int, b: int) -> tuple[int, int]:
    prev_r, r = a, b
    prev_x, x = 1, 0
    prev_y, y = 0, 1

    while r != 0:
        q = prev_r // r

        prev_r, r = r, prev_r - q * r
        prev_x, x = x, prev_x - q * x
        prev_y, y = y, prev_y - q * y

    return prev_x, prev_y


def bezout(integers: list[int]) -> list[int]:
    a, b = integers[0], integers[1]
    x, y = _bezout(a, b)

    coeffs: list[int] = [x, y]
    for i in range(2, len(integers)):
        a = a * x + b * y
        b = integers[i]
        x, y = _bezout(a, b)

        for j in range(i):
            coeffs[j] *= x

        coeffs.append(y)

    return coeffs


def qr(A, mode="economic"):
    # custom qr decomposition such that r is lower triangular
    A_rev = np.fliplr(A)
    q_rev, r_rev = la.qr(A_rev, mode=mode)
    q = np.fliplr(q_rev)
    r = np.flipud(np.fliplr(r_rev))

    return q, r


def lattice_projection(c_p):
    n = len(c_p)
    q = c_p.copy()

    x_p = np.zeros(n, dtype=int)
    w_p = np.zeros(n - 1, dtype=int)
    g = np.zeros(n - 1, dtype=int)

    w_p[0] = 1  # for convenience
    g[0] = 1  # q is assumed to be coprime

    for i in range(len(q) - 2):
        q[i:] = q[i:] // g[i]
        g[i + 1] = math.gcd(*q[i + 1 :])
        x_p[i], w_p[i + 1] = _bezout(q[i], g[i + 1])

    q[-2:] = q[-2:] // g[-1]
    x_p[-2:] = _bezout(*q[-2:])  # last two particular solutions

    # building w
    prod = np.cumprod(w_p)
    w = x_p * np.hstack((prod, prod[-1]))

    # building hessenberg matrix H
    # TODO: there must be a better way to do this
    H = np.zeros((n, n - 1))

    for i in range(n - 2):
        for j in range(i):
            H[i, j] = q[j] * x_p[i] * np.prod(w_p[j + 2 : i + 1])
        H[i, i] = g[i + 1]

    # last two rows
    for i in (-2, -1):
        for j in range(n - 2):
            H[i, j] = -q[j] * x_p[i] * np.prod(w_p[j + 2 : -1])
    H[-2, -1] = q[-1]
    H[-1, -1] = -q[-2]

    return H, w


def solve_lattice(c_p, A, b, u):
    n = len(c_p)
    q, r = qr(A, mode="economic")
    h, w = lattice_projection(c_p)

    # transformed constraints
    M = r @ h
    rhs_1 = q.T @ b
    rhs_2 = r @ w

    mask = np.ones(n, dtype=bool)
    for k in range(u, -1, -1):
        rhs = rhs_1 - k * rhs_2

        for i in range(n - 1, -1, -1):
            mask[min(i + 1, n - 1)] = True
            mask[i] = False

            try:
                params = la.solve_triangular(M[mask, :], rhs[mask], lower=True)
            except np.linalg.LinAlgError:
                continue

            # check params are integers
            if not np.allclose(params, params.astype(int), rtol=1e-5):
                continue

            # check the non-active constraint is satisfied
            print(params)
            print(M[~mask] @ params, "\t", rhs[~mask])
            print("x: ", k * w + h @ params)
            if np.allclose(M[~mask] @ params, rhs[~mask]):
                x = k * w + h @ params
                return x

        mask[0] = True

    else:
        print("problem is infeasible")
        return None


def solve_feasibility_naive(c_p, A, b, u):
    import pulp as pl

    m, n = A.shape
    h, w = lattice_projection(c_p)
    print(w)

    lhs = A @ h
    rhs_1 = b 
    rhs_2 = A @ w

    # build model and do initial solve
    prob = pl.LpProblem("feasibility problem")
    t = [pl.LpVariable(f"t_{j}", cat="Integer") for j in range(n - 1)]

    for i in range(m):
        prob += pl.lpDot(lhs[i], t) <= rhs_1[i] - u * rhs_2[i]

    solver = pl.PULP_CBC_CMD(msg=False)
    status = solver.solve(prob)

    # solve with warmed state
    k = u
    while status == -1:
        k -= 1
        print(f"on c-layer = {k}")

        # update model solution
        for i in range(m):
            prob.constraints[f"_C{i + 1}"].changeRHS(rhs_1[i] - k * rhs_2[i])
        status = solver.solve(prob)

        if (k == 0) and (status == -1):
            msg = "problem is infeasible"
            raise ValueError(msg)


    params = np.array([t_i.varValue for t_i in t])
    return h @ params + k * w


p = np.array([9.913, 9.891, 9.514, 9.379])

# diophantine prices (p == p_1 / p_2)
p_1 = np.array([9913, 9891, 9514, 9379])
p_2 = np.ones_like(p_1) * 1_000

# coprime multiple of p
m = math.lcm(*p_2)
c_p = p_1 * (m // p_2)
c_p = c_p // np.dot(c_p, bezout(c_p))

s = sum(p) - 1
num_layers = math.floor(s * c_p[-1] / p[-1])
print(f"{s=}")
print(f"{num_layers=}")

# constraints
n = len(p)
A = -np.eye(n)
b = np.zeros(n)

x = solve_feasibility_naive(c_p, A, b, num_layers)
print(f"solution: {x}")
print(f"objective: {np.dot(x, p)}")
