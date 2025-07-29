import os
import re
import math
import tempfile as tmp
import pulp as lp
import numpy as np

from .schema import StatsSchema
from .decorators import repeat


def solve_pulp(p, rhs, num_reps, **kwargs):
    # create lp problem
    x = [lp.LpVariable(f"b_{i}", lowBound=0, cat=lp.LpInteger) for i in range(len(p))]
    prob = lp.LpProblem(sense=lp.LpMaximize)
    prob += lp.lpDot(p, x), "objective"
    prob += lp.lpDot(p, x) <= rhs, "constraint"

    @repeat(num_reps)
    def solve_pulp():
        prob.solve(lp.PULP_CBC_CMD(**kwargs))
        return np.array([lp.value(var) for var in x])

    x, (mu, sigma) = solve_pulp()

    # pulp does not provide a default method to determine
    # whether an instance was stopped due to a timeout
    pattern = r"Result - Stopped on time limit"
    string = open(kwargs["logPath"]).read()
    timed_out = bool(re.search(pattern, string))

    return x, (mu, sigma), timed_out


def stats_dioph_as_rhs_increases(solver, q, m, rhs):
    stats = StatsSchema(solver.__name__)

    for i, u in enumerate(rhs):
        eta = math.floor(u / m)
        x, (mu, sigma), timed_out = solver(q, eta)

        # sanity checks
        if np.any(x < 0):
            print("[ERROR]: non-negativity constraint violated")
        if np.dot(q, x) != eta:
            print("[ERROR]: dioph equation is not satisfied")

        # update
        stats.update(mu, sigma, m * eta, u, timed_out)

    return stats


def stats_bb_as_rhs_increases(p, rhs, num_reps, **kwargs):
    stats = StatsSchema(kwargs.pop("name"))

    base_path = os.path.join("/tmp", kwargs.pop("basePath"), stats.solver)
    os.makedirs(base_path, exist_ok=True)

    for i, u in enumerate(rhs):
        _, log_path = tmp.mkstemp(dir=base_path)
        kwargs["logPath"] = log_path
        x, (mu, sigma), timed_out = solve_pulp(p, u, num_reps, **kwargs)

        # update
        stats.update(mu, sigma, np.dot(p, x), u, timed_out)

    return stats
