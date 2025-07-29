import os
import re
import math
import logging
import pulp as lp
import numpy as np
import json
import multiprocessing as mp

from shutil import rmtree
from tempfile import mkstemp
from dataclasses import asdict

from .utils import setup_logger
from .constants import NUM_REPS, bb_raw_options, bb_full_options
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

    if timed_out:
        logging.warn("solver execution timed out")

    return x, (mu, sigma), timed_out


def stats_dioph_as_rhs_increases(solver, q, m, rhs):
    stats = StatsSchema(solver.__name__)

    for u in rhs:
        eta = math.floor(u / m)
        x, (mu, sigma), timed_out = solver(q, eta)

        # sanity checks
        if np.any(x < 0):
            logging.error("non-negativity constraint violated")
        if np.dot(q, x) != eta:
            logging.error("dioph equation is not satisfied")

        # update
        stats.update(mu, sigma, (m * eta).item(), u.item(), timed_out)

    return stats


def stats_bb_as_rhs_increases(p, rhs, num_reps, **kwargs):
    stats = StatsSchema(kwargs.pop("name"))

    base_path = os.path.join(".", kwargs.pop("logPath"), stats.solver)
    os.makedirs(base_path, exist_ok=True)
    logging.info(f"created {base_path} to store temp solver log info")

    for u in rhs:
        _, log_path = mkstemp(dir=base_path)
        kwargs["logPath"] = log_path
        x, (mu, sigma), timed_out = solve_pulp(p, u, num_reps, **kwargs)

        # update
        stats.update(mu, sigma, np.dot(p, x).item(), u.item(), timed_out)

    return stats


def worker_dioph_as_rhs_increases(job_id, log_path, solver, q, m, rhs):
    save_path = os.path.join(log_path, "stats.json")

    setup_logger(job_id, log_path)
    logging.info(f"saving log info on {log_path}")

    logging.info("starting work")
    stats = stats_dioph_as_rhs_increases(solver, q, m, rhs)
    logging.info(f"finished work. saving stats info on {save_path}")

    with open(save_path, "w") as outfile:
        json.dump(asdict(stats), outfile)

    logging.info("done. terminating process")


def worker_bb_as_rhs_increases(job_id, log_path, p, rhs, num_reps, **kwargs):
    save_path = os.path.join(log_path, "stats.json")

    setup_logger(job_id, log_path)
    logging.info(f"saving log info on {log_path}")

    logging.info("starting work")
    stats = stats_bb_as_rhs_increases(p, rhs, num_reps, **kwargs)
    logging.info(f"finished work. saving stats info on {save_path}")

    with open(save_path, "w") as outfile:
        json.dump(asdict(stats), outfile)

    logging.info("done. terminating process")


def experiment_rhs(p, q, m, rhs, solver, path, num_reps=NUM_REPS):
    processes = []
    num_experiments, dim = p.shape

    logging.info(f"launching {3 * num_experiments} processes")
    for i in range(num_experiments):
        dioph_process = mp.Process(
            target=worker_dioph_as_rhs_increases,
            args=(
                10 * i,
                os.path.join(path, "dioph", f"{dim}n-{i}w"),
                solver,
                q[i].copy(),
                m[i],
                rhs.copy(),
            ),
        )
        bb_raw_process = mp.Process(
            target=worker_bb_as_rhs_increases,
            args=(
                10 * i + 1,
                os.path.join(path, "bb_raw", f"{dim}n-{i}w"),
                p[i].copy(),
                rhs.copy(),
                num_reps,
            ),
            kwargs=bb_raw_options.copy(),
        )
        bb_full_process = mp.Process(
            target=worker_bb_as_rhs_increases,
            args=(
                10 * i + 2,
                os.path.join(path, "bb_full", f"{dim}n-{i}w"),
                p[i].copy(),
                rhs.copy(),
                num_reps,
            ),
            kwargs=bb_full_options.copy(),
        )

        # start processes and add to running processes
        dioph_process.start()
        bb_raw_process.start()
        bb_full_process.start()
        processes.extend([dioph_process, bb_raw_process, bb_full_process])

    for process in processes:
        process.join()

    # cleanup temp log files
    logging.info("deleting temp solver log files")
    rmtree(os.path.join(bb_raw_options["logPath"], bb_raw_options["name"]))
    rmtree(os.path.join(bb_full_options["logPath"], bb_full_options["name"]))
