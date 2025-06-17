import math
import pulp
import numpy as np

from time import perf_counter_ns


# math module has not yet implemented a sign function
# see (https://bugs.python.org/msg59154)
sign = lambda x: int(math.copysign(1.0, x))  # noqa: E731


def _bezout(a: int, b: int) -> tuple[int, int]:
    sgn_a, sgn_b = sign(a), sign(b)
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


def feasible_point_finite(c_p, eta):
    if len(c_p) == 2:
        x_prime, y_prime = _bezout(c_p[0], c_p[1])
        lb = math.ceil(-eta  * x_prime / c_p[1])
        ub = math.floor(eta * y_prime / c_p[0])

        if ub < lb:
            return None

        x = eta * x_prime + lb * c_p[1]
        y = eta * y_prime - lb * c_p[0]
        return (x.item(), y.item())

    g = math.gcd(*c_p[1:])
    x_prime, omega_prime = _bezout(c_p[0], g)

    # bounds for feasible parameters
    lb = math.ceil(-eta * x_prime / g)
    ub = math.floor(eta * omega_prime / c_p[0])

    for t in range(lb, ub + 1):
        omega = eta * omega_prime - t * c_p[0]
        x = eta * x_prime + t * g
        rest = feasible_point_finite((c_p[1:] / g).astype(int), omega)

        if rest is not None:
            return (x, *rest)
    return None


def feasible_point_infinite(c_p, eta):
    if len(c_p) == 2:
        x_prime, y_prime = _bezout(c_p[0], c_p[1])

        # cases for bounds depending on the signs of c_p
        bound_1 = -eta * x_prime / c_p[1]
        bound_2 = eta * y_prime / c_p[0]
        lb = -math.inf
        ub = math.inf

        if (c_p[0] >= 0) and (c_p[1] >= 0):
            lb = math.ceil(bound_1)
            ub = math.floor(bound_2)
            t = lb

        elif (c_p[0] <= 0) and (c_p[1] <= 0):
            lb = math.ceil(bound_2)
            ub = math.floor(bound_1)
            t = lb

        elif (c_p[0] <= 0) and (c_p[1] >= 0):
            lb = math.ceil(max(bound_1, bound_2))
            t = lb

        elif (c_p[0] >= 0) and (c_p[1] <= 0):
            ub = math.floor(min(bound_1, bound_2))
            t = ub
        else:
            msg = "trichotomy fails. this should never happen"
            raise AssertionError(msg)

        if ub < lb:
            return None

        x = eta * x_prime + t * c_p[1]
        y = eta * y_prime - t * c_p[0]
        return (x.item(), y.item())

    g = math.gcd(*c_p[1:])
    x_prime, omega_prime = _bezout(c_p[0], g)

    # only lower bounds are needed
    lb = math.ceil(-eta * x_prime / g)
    t = lb

    while True:
        omega = eta * omega_prime - t * c_p[0]
        x = eta * x_prime + t * g
        rest = feasible_point_infinite((c_p[1:] / g).astype(int), omega)

        if rest is not None:
            return (x, *rest)

        t += 1


def solve_dioph(c_p, k, s, num_iter=100):
    times = np.zeros(num_iter)
    eta = math.floor(s / k)

    for i in range(num_iter):
        start = perf_counter_ns()

        if np.all(c_p >= 0.0):
            # problem is infeasible
            if s < 0:
                x = None

            # finite case: eta layer may not contain feasible points
            while eta >= 0:
                if (x := feasible_point_finite(c_p, eta)) is not None:
                    x = np.asarray(x)
                    break
                eta -= 1
        else:
            # infinite case: eta layer contains feasible points
            x = feasible_point_infinite(c_p, eta)
            x = np.asarray(x)

        times[i] = perf_counter_ns() - start
        print(".", end="")
    print()

    # report stats
    mu_ns = times.mean()
    std_ns = times.std(ddof=1)
    print(f"[INFO]: took {mu_ns * 1e-6:0.4f} ± {std_ns * 1e-6:0.4f} ms to solve")

    return x, (mu_ns, std_ns)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    FILENAME = "3d-4dig"
    NUM_ITER = 5

    p = np.array([9.9314, 9.7752, 9.5358, 9.1353])

    # finding coprime multiple of projectively rational vector
    num_decimals = 4
    p_1 = np.array([99314, 97752, 95358, 91353])
    g = math.gcd(*p_1)
    c_p = p_1 // g
    multiplier = g / math.pow(10, num_decimals)
    print(f"{multiplier=}")
    np.testing.assert_almost_equal(p, multiplier * c_p)

    slack = np.logspace(2, 4, num=100, base=10.0, dtype=int)
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
        print(f"{pulp_sol=}")

        # diophantine method
        print("[INFO]: solving with diophantine")
        dioph_sol, (mu, std) = solve_dioph(c_p, multiplier, s, num_iter=NUM_ITER)
        stats["dioph"]["mean"][i] = mu
        stats["dioph"]["std"][i] = std
        print(f"{dioph_sol=}")

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
    plt.show()

    # plt.savefig(f"figs/{FILENAME}.pdf")
    # with open(f"times/{FILENAME}.json", "w") as outfile:
        # json.dump(stats, outfile)
