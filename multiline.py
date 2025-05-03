import math
import pulp
import numpy as np

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


def feasible_point(c_p, k):
    n = len(c_p)
    a = c_p[0] if n == 2 else math.gcd(*c_p[:-1])
    b = c_p[-1]
    g = math.gcd(a, b)

    if k % g != 0:
        print("[ERROR]: problem is infeasible")
        return None

    k = k // g
    a = a // g
    b = b // g
    x, y = _bezout(a, b)

    # bounds for integer parameters
    lb = math.ceil(-k * x / b)
    ub = math.floor(k * y / a)

    if ub < lb:
        return None
    elif n == 2:
        x_1 = k * x + b * lb
        x_2 = k * y - a * lb
        return [x_1.item(), x_2.item()]

    for t in range(lb, ub + 1):
        w = k * x + b * t
        point = feasible_point(c_p[:-1], w)

        if point is not None:
            x_n = k * y - a * t
            return [*point, x_n.item()]

    return None


def solve_pulp(p, s, num_iter=100):
    # create lp problem
    x = [
        pulp.LpVariable(f"b_{i}", lowBound=0, cat=pulp.LpInteger) for i in range(len(p))
    ]
    prob = pulp.LpProblem(sense=pulp.LpMaximize)
    prob += pulp.lpDot(p, x), "objective"
    prob += pulp.lpDot(p, x) <= s, "constraint"

    # solve problem
    times = np.zeros(num_iter)
    for i in range(num_iter):
        start = perf_counter_ns()
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        times[i] = perf_counter_ns() - start
        print(".", end="")
    print()

    # report stats
    mu_ns = times.mean()
    std_ns = times.std(ddof=1)
    print(f"[INFO]: took {mu_ns * 1e-6:0.4f} ± {std_ns * 1e-6:0.4f} ms to solve")

    res = np.array([pulp.value(var) for var in x])
    return res, (mu_ns, std_ns)


def solve_dioph(c_p, p, s, num_iter=100):
    times = np.zeros(num_iter)

    num_layers = math.floor(s * c_p[-1] / p[-1])
    for i in range(num_iter):
        start = perf_counter_ns()
        for k in range(num_layers, 0, -1):
            res = feasible_point(c_p, k)
            if res is not None:
                break
        else:
            print(f"[ERROR:] could not find solution with slack {s}")
            res = np.zeros_like(c_p)
        times[i] = perf_counter_ns() - start
        print(".", end="")
    print()

    # report stats
    mu_ns = times.mean()
    std_ns = times.std(ddof=1)
    print(f"[INFO]: took {mu_ns * 1e-6:0.4f} ± {std_ns * 1e-6:0.4f} ms to solve")

    return np.asarray(res), (mu_ns, std_ns)


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    FILENAME = "3d-4dig"
    NUM_ITER = 50

    p = np.array([9.9314, 9.7752, 9.5358])

    # equivalent diophantine prices (p == p_1 / p_2)
    p_1 = np.array([99314, 97752, 95358])
    p_2 = np.ones_like(p_1) * 10_000

    # finding coprime multiple of the projectively rational vector
    m = math.lcm(*p_2)
    c_p = p_1 * (m // p_2)
    c_p = c_p // np.dot(c_p, bezout(c_p))
    print(c_p / p)

    c_p = p_1 * (m // p_2)
    g = np.dot(c_p, bezout(c_p))
    print(m / g)
    exit(0)

    slack = np.logspace(2, 7, num=100, base=10.0, dtype=int)
    stats: dict[str, dict[str, list[float]]] = {
        "pulp": {
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

        # pulp method
        print("[INFO]: solving with pulp")
        pulp_sol, (mu, std) = solve_pulp(p, s, num_iter=NUM_ITER)
        stats["pulp"]["mean"][i] = mu
        stats["pulp"]["std"][i] = std

        # diophantine method
        print("[INFO]: solving with diophantine")
        dioph_sol, (mu, std) = solve_dioph(c_p, p, s, num_iter=NUM_ITER)
        stats["dioph"]["mean"][i] = mu
        stats["dioph"]["std"][i] = std

        # cannot guarantee both solutions to be equal
        dioph_obj = np.dot(p, dioph_sol)
        pulp_obj = np.dot(p, pulp_sol)

        if not np.isclose(dioph_obj, pulp_obj):
            print(f"[ERROR]: objective values don't match for slack {s}")
            print(f"[DEBUG]: {pulp_sol=} -> obj = {dioph_obj}")
            print(f"[DEBUG]: {dioph_sol=} -> obj = {pulp_obj}")
            print(f"[DEBUG]: dioph_obj > pulp_obj is {dioph_obj > pulp_obj}")
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

    plt.savefig(f"figs/{FILENAME}.pdf")
    with open(f"times/{FILENAME}.json", "w") as outfile:
        json.dump(stats, outfile)
