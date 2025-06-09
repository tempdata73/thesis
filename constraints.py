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


def solve_feasibility_naive(c_p, A, b, u):
    import pulp as pl

    m, n = A.shape
    h, w = lattice_projection(c_p)

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


def solve_feasibility_backtrack(c_p, A, b, u, depth=1):
    print(f"\n{depth=}")
    print(f"cost vector: {c_p}")
    print(f"{u=}")

    if len(c_p) == 2:
        x_p, y_p = _bezout(c_p[0], c_p[1])
        lb = math.ceil(-u * y_p / c_p[0])
        ub = math.floor(u * x_p / c_p[1])

        print("on base case")
        print(f"cost vector: {c_p}")
        print(f"particular solutions: ({x_p}, {y_p})")
        print(f"feasible interval: ({lb}, {ub})")

        for t in range(lb, ub + 1):
            x = u * x_p - t * c_p[1]
            y = u * y_p + t * c_p[0]
            lhs = A[:, 0] * x + A[:, 1] * y

            if np.all(lhs <= b):
                print(f"solution: (x, y) = ({x}, {y})")
                return [x, y]

        return None

    g_next = math.gcd(*c_p[1:])
    x_p, w_p = _bezout(c_p[0], g_next)
    lb = math.ceil(-u * w_p / c_p[0])
    ub = math.floor(u * x_p / g_next)

    print(f"particular solutions: (x_p, w_p) = ({x_p}, {w_p})")
    print(f"feasible interval: ({lb}, {ub})")

    for t in range(lb, ub + 1):
        x = u * x_p - t * g_next
        w = u * w_p + t * c_p[0]
        rhs = b - A[:, 0] * x

        print(f"on parameter {t=}")
        print(f"solution: (x, w) = ({x}, {w})")

        x_rest = solve_feasibility_backtrack(
            c_p[1:] // g_next, A[:, 1:], rhs, w, depth + 1
        )

        if x_rest is not None:
            return [x, *x_rest]

        print(f"\nbacktracking: {depth=}")
    return None


def solve_lattice(c_p, A, b, u):
    print(f"ON C-LAYER: {u}\n")
    while (x := solve_feasibility_backtrack(c_p, A, b, u)) is None:
        if u == 0:
            msg = "problem is infeasible"
            raise ValueError(msg)
        print(f"ON C-LAYER: {u}\n")
        u -= 1
    return x


p = np.array([9.982, 9.990, 9.113, 9.313])

# diophantine prices (p == p_1 / p_2)
p_1 = np.array([9982, 9990, 9113, 9313])
p_2 = np.ones_like(p_1) * 1_000

# coprime multiple of p
m = math.lcm(*p_2)
c_p = p_1 * (m // p_2)
c_p = c_p // np.dot(c_p, bezout(c_p))

s = sum(p) - 1
num_layers = math.ceil(s * c_p[-1] / p[-1])
print(f"{s=}")
print(f"{num_layers=}\n")

# constraints
n = len(p)
A = np.eye(n)
b = np.ones(n)

x = solve_lattice(c_p, A, b, num_layers)
x = np.asarray(x, dtype=int)

# x = solve_feasibility_naive(c_p, A, b, num_layers)
print()
print(f"solution: {x}")
print(f"objective: {np.dot(x, p)}")
