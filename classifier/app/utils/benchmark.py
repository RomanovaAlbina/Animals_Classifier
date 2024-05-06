import functools
import logging
import time


def timer(func):
    @functools.wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_ts = time.time()
        result = func(*args, **kwargs)
        end_ts = time.time()

        logging.info(f"{func.__name__} executed in {(end_ts - start_ts):.4f}s")
        return result

    return timer_wrapper
