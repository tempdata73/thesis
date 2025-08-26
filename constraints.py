import math
import numpy as np
import pulp as pl
import scipy.linalg as la

from time import perf_counter_ns


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
    x = h @ params + k * w
    return x


def solve_feasibility_backtrack(c_p, A, b, u):
    if len(c_p) == 2:
        x_p, y_p = _bezout(c_p[0], c_p[1])
        lb = math.ceil(-u * y_p / c_p[0])
        ub = math.floor(u * x_p / c_p[1])

        for t in range(lb, ub + 1):
            x = u * x_p - t * c_p[1]
            y = u * y_p + t * c_p[0]
            lhs = A[:, 0] * x + A[:, 1] * y

            if np.all(lhs <= b):
                return [x, y]

        return None

    g_next = math.gcd(*c_p[1:])
    x_p, w_p = _bezout(c_p[0], g_next)
    lb = math.ceil(-u * w_p / c_p[0])
    ub = math.floor(u * x_p / g_next)

    for t in range(lb, ub + 1):
        x = u * x_p - t * g_next
        w = u * w_p + t * c_p[0]
        rhs = b - A[:, 0] * x

        # unpack previous solutions
        x_rest = solve_feasibility_backtrack(c_p[1:] // g_next, A[:, 1:], rhs, w)
        if x_rest is not None:
            return [x, *x_rest]

    return None


def solve_lattice(c_p, A, b, u, num_iter=50):
    times = np.zeros(num_iter)

    for i in range(num_iter):
        start = perf_counter_ns()
        while (x := solve_feasibility_backtrack(c_p, A, b, u)) is None:
            if u == 0:
                msg = "problem is infeasible"
                raise ValueError(msg)
            u -= 1
        times[i] = perf_counter_ns() - start
        print(".", end="")
    print()

    # report stats
    mu_ns = times.mean()
    std_ns = times.std(ddof=1)
    print(f"[INFO]: took {mu_ns * 1e-6:0.4f} ± {std_ns * 1e-6:0.4f} ms to solve")

    return np.asarray(x), (mu_ns, std_ns)


def solve_pulp(p, A, b, u, num_iter=50):
    x = [pl.LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(len(p))]
    prob = pl.LpProblem(sense=pl.LpMaximize)
    prob += pl.lpDot(p, x)
    prob += pl.lpDot(p, x) <= u

    # custom constraints
    for i in range(len(A)):
        prob += pl.lpDot(A[:, i], x) <= b[i]

    # solve problem
    times = np.zeros(num_iter)
    for i in range(num_iter):
        start = perf_counter_ns()
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        times[i] = perf_counter_ns() - start
        print(".", end="")
    print()

    # report stats
    mu_ns = times.mean()
    std_ns = times.std(ddof=1)
    print(f"[INFO]: took {mu_ns * 1e-6:0.4f} ± {std_ns * 1e-6:0.4f} ms to solve")

    res = np.array([pl.value(var) for var in x])
    return res, (mu_ns, std_ns)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    NUM_ITER = 10

    p = np.array([9.9314, 9.7752, 9.5358])
    # equivalent diophantine prices (p == p_1 / p_2)
    p_1 = np.array([99314, 97752, 95358])
    p_2 = np.ones_like(p_1) * 10_000

    # finding coprime multiple of the projectively rational vector
    m = math.lcm(*p_2)
    c_p = p_1 * (m // p_2)
    c_p = c_p // np.dot(c_p, bezout(c_p))

    slack = np.logspace(2, 4, num=100, base=10.0, dtype=int)

    stats: dict[str, dict[str, list[float]]] = {
        "pulp": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
        "naive": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
        "dioph": {
            "mean": [0.0] * len(slack),
            "std": [0.0] * len(slack),
        },
    }

    for i, s in enumerate(slack):
        print(f"[INFO]: on problem {i + 1} with slack {s}")

        # adding (simple) constraints to problem: p[i] <= 0.7 * s
        n = len(p)
        A = np.eye(n)
        b = np.ones(n) * s * 0.7
        num_layers = math.floor(s * c_p[-1] / p[-1])

        # pulp method
        print("[INFO]: solving with pulp")
        pulp_sol, (mu, std) = solve_pulp(p, A, b, s, num_iter=NUM_ITER)
        stats["pulp"]["mean"][i] = mu
        stats["pulp"]["std"][i] = std

        # diophantine method
        print("[INFO]: solving with diophantine")
        dioph_sol, (mu, std) = solve_lattice(c_p, A, b, num_layers, num_iter=NUM_ITER)
        stats["dioph"]["mean"][i] = mu
        stats["dioph"]["std"][i] = std

        # cannot guarantee both solutions to be equal
        dioph_obj = np.dot(p, dioph_sol)
        pulp_obj = np.dot(p, pulp_sol)

        if not np.isclose(dioph_obj, pulp_obj):
            print(f"[ERROR]: objective values don't match for slack {s}")
            print(f"[DEBUG]: {pulp_sol=} -> obj = {pulp_obj}")
            print(f"[DEBUG]: {dioph_sol=} -> obj = {dioph_obj}")
            print(f"[DEBUG]: dioph_obj > pulp_obj is {dioph_obj > pulp_obj}")
            raise ValueError()
        print()

    # plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(slack, stats["pulp"]["mean"], "o--", c="firebrick", zorder=5, label="b&b")
    ax.loglog(
        slack, stats["dioph"]["mean"], "o--", c="navy", zorder=5, label="diophantine"
    )
    ax.grid()

    ax.set_ylabel("time [ns]")
    ax.set_xlabel("slack")
    ax.legend()
    plt.show()
