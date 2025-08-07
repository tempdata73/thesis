import os
import math
import re
import logging
import pulp as lp
import numpy as np
import json
import multiprocessing as mp

from tempfile import mkstemp
from dataclasses import asdict

from .constants import NUM_REPS, RANDOM_SEED
from .utils import setup_logger
from .schema import StatsSchema
from .decorators import repeat
from .knapsack import ukp_dp


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

    if timed_out:
        logging.warn("solver execution timed out")

    return x, (mu, sigma), timed_out


def stats_dioph_as_rhs_increases(solver, q, m, rhs):
    stats = StatsSchema(solver.__name__)

    for i, u in enumerate(rhs):
        logging.info(f"on problem {i + 1} of {len(rhs)}")
        eta = math.floor(u / m)
        x, (mu, sigma), timed_out = solver(q, eta)

        # in case all evaluations timed out
        if x is None:
            logging.info("all functions evaluations timed out")
            x = np.zeros_like(q)

        # sanity checks
        if np.any(x < 0):
            logging.error("non-negativity constraint violated")
        if np.dot(q, x) != eta:
            obj = np.dot(q, x)
            logging.error(f"dioph equation is not satisfied. {obj=} is not {eta=}")

        # update
        stats.update(mu, sigma, (m * eta).item(), u.item(), timed_out)

    return stats


def stats_dioph_as_dim_increases(solver, dims, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed=seed)
    stats = StatsSchema(solver.__name__)

    for i, dim in enumerate(dims):
        logging.info(f"on problem {i + 1} of {len(dims)}")
        p = np.sort(rng.integers(10, 5 * dim, size=dim))[::-1]
        m = math.gcd(*p)
        q = p // m

        if not np.allclose(p, m * q):
            logging.error("could not determine coprime multiple. skipping problem")
            continue

        k = 0.5 if dim <= 20_000 else 0.1
        u = (p.sum() * k).astype(int)
        eta = math.floor(u / m)
        x, (mu, sigma), timed_out = solver(q, eta)

        # in case all evaluations timed out
        if x is None:
            logging.info("all functions evaluations timed out")
            x = np.zeros_like(q)

        # sanity checks
        if np.any(x < 0):
            logging.error("non-negativity constraint violated")

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


def stats_bb_as_rhs_increases(p, rhs, num_reps=NUM_REPS, **kwargs):
    stats = StatsSchema(kwargs.pop("name"))

    base_path = os.path.join(".", kwargs.pop("logPath"), stats.solver)
    os.makedirs(base_path, exist_ok=True)
    logging.info(f"created {base_path} to store temp solver log info")

    for i, u in enumerate(rhs):
        logging.info(f"on problem {i + 1} of {len(rhs)}")

        _, log_path = mkstemp(dir=base_path)
        kwargs["logPath"] = log_path
        x, (mu, sigma), timed_out = solve_pulp(p, u, num_reps, **kwargs)

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


def stats_bb_as_dim_increases(dims, num_reps=NUM_REPS, seed=RANDOM_SEED, **kwargs):
    rng = np.random.default_rng(seed=seed)
    stats = StatsSchema(kwargs.pop("name"))

    base_path = os.path.join(".", kwargs.pop("logPath"), stats.solver)
    os.makedirs(base_path, exist_ok=True)
    logging.info(f"created {base_path} to store temp solver log info")

    for i, dim in enumerate(dims):
        logging.info(f"on problem {i + 1} of {len(dims)}")

        p = rng.integers(10, 5 * dim, size=dim)
        k = 0.5 if dim <= 20_000 else 0.1
        u = (p.sum() * k).astype(int)

        _, log_path = mkstemp(dir=base_path)
        kwargs["logPath"] = log_path
        x, (mu, sigma), timed_out = solve_pulp(p, u, num_reps, **kwargs)

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


# NOTE: in case i need to use other solvers (e.g. mtu),
# make this more generic instead of creating another function
def stats_dp_as_rhs_increases(p, rhs):
    stats = StatsSchema("dp")

    for i, u in enumerate(rhs):
        logging.info(f"on problem {i + 1} of {len(rhs)}")
        x, (mu, sigma), timed_out = ukp_dp(p, u)

        # in case all evaluations timed out
        if x is None:
            logging.info("all functions evaluations timed out")
            x = np.zeros_like(p)

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


def stats_dp_as_dim_increases(dims, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed=seed)
    stats = StatsSchema("dp")

    for i, dim in enumerate(dims):
        logging.info(f"on problem {i + 1} of {len(dims)}")

        p = np.sort(rng.integers(10, 5 * dim, size=dim))
        k = 0.5 if dim <= 20_000 else 0.1
        u = (p.sum() * k).astype(int)
        x, (mu, sigma), timed_out = ukp_dp(p, u)

        # in case all evaluations timed out
        if x is None:
            logging.info("all functions evaluations timed out")
            x = np.zeros_like(p)

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


def task(job):
    log_path = job["log_path"]
    setup_logger(log_path)
    logging.info(f"saving log info on {log_path}")

    logging.info("starting work")
    stats = job["job"]()

    save_path = os.path.join(log_path, "stats.json")
    logging.info(f"finished work. saving stats info on {save_path}")
    with open(save_path, "w") as outfile:
        json.dump(asdict(stats), outfile)

    logging.info("done. terminating process")


def run_parallel(jobs):
    processes = []

    logging.info(f"launching {len(jobs)} processes")
    for job in jobs:
        process = mp.Process(target=task, args=(job,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
