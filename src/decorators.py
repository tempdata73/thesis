import signal
import logging
import statistics as stats

from time import perf_counter_ns
from functools import wraps


from .constants import NUM_REPS, TIMEOUT


def alarm_handler(signum, frame):
    msg = "function execution timed out."
    raise TimeoutError(msg)


def repeat_with_timeout(num_reps=NUM_REPS, timeout=TIMEOUT):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"running {func.__name__} {num_reps} times")
            signal.signal(signal.SIGALRM, alarm_handler)

            times = [0 for _ in range(num_reps)]
            timeout_counter = 0

            # get samples. even if function timed out,
            # the observation will still be part of the sample
            prev_res = None
            for i in range(num_reps):
                signal.alarm(timeout)
                start = perf_counter_ns()

                try:
                    res = func(*args, **kwargs)
                    prev_res = res
                except TimeoutError as e:
                    logging.warn(f"alarm_handler: {e}")
                    timeout_counter += 1
                    res = prev_res
                    continue
                finally:
                    signal.alarm(0)
                    times[i] = perf_counter_ns() - start

            # calc stats
            mu = stats.fmean(times)
            std = stats.stdev(times, xbar=mu)
            logging.info(f"took {mu * 1e-6:0.4f} ± {std * 1e-6:0.4f} ms")

            # majority rule to determine whether the function times out in general
            timed_out = (timeout_counter / num_reps) >= 0.5

            return res, (mu, std), timed_out

        return wrapper

    return decorator


def repeat(num_reps=NUM_REPS):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"running {func.__name__} {num_reps} times")
            times = [0 for _ in range(num_reps)]

            # get samples
            for i in range(num_reps):
                start = perf_counter_ns()
                res = func(*args, **kwargs)
                times[i] = perf_counter_ns() - start

            # calc stats
            mu = stats.fmean(times)
            std = stats.stdev(times, xbar=mu)
            logging.info(f"took {mu * 1e-6:0.4f} ± {std * 1e-6:0.4f} ms")
            return res, (mu, std)

        return wrapper

    return decorator
