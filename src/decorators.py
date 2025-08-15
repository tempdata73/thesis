import signal
import logging
import statistics as stats

from time import perf_counter_ns
from functools import wraps


from .constants import NUM_REPS, TIMEOUT, NUM_IGNORE


def alarm_handler(signum, frame):
    msg = "function execution timed out."
    raise TimeoutError(msg)


def repeat_with_timeout(num_reps=NUM_REPS, timeout=TIMEOUT, num_ignore=NUM_IGNORE):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"running {func.__name__} {num_reps + num_ignore} times")
            logging.info(f"ignoring first {num_ignore} runs")

            signal.signal(signal.SIGALRM, alarm_handler)
            times = []
            timeout_counter = 0

            # get samples. even if function timed out,
            # the observation will still be part of the sample
            prev_res = None
            for i in range(num_reps + num_ignore):
                signal.alarm(timeout)
                start = perf_counter_ns()

                try:
                    res = func(*args, **kwargs)
                    prev_res = res
                except TimeoutError as e:
                    logging.warn(f"alarm_handler: {e}")
                    timeout_counter += 1
                    res = prev_res
                finally:
                    signal.alarm(0)
                    times.append(perf_counter_ns() - start)

            # calc stats
            mu = stats.fmean(times[num_ignore:])
            std = stats.stdev(times[num_ignore:], xbar=mu)
            logging.info(f"took {mu * 1e-6:0.4f} ± {std * 1e-6:0.4f} ms")

            # majority rule to determine whether the function times out in general
            timed_out = (timeout_counter / (num_reps + num_ignore)) >= 0.5

            return res, (mu, std), timed_out

        return wrapper

    return decorator


def repeat(num_reps=NUM_REPS, num_ignore=NUM_IGNORE):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"running {func.__name__} {num_reps} times")
            logging.info(f"running {func.__name__} {num_reps + num_ignore} times")
            times = []

            # get samples
            for i in range(num_reps + num_ignore):
                start = perf_counter_ns()
                res = func(*args, **kwargs)
                times.append(perf_counter_ns() - start)

            # calc stats
            mu = stats.fmean(times[num_ignore:])
            std = stats.stdev(times[num_ignore:], xbar=mu)
            logging.info(f"took {mu * 1e-6:0.4f} ± {std * 1e-6:0.4f} ms")
            return res, (mu, std)

        return wrapper

    return decorator
