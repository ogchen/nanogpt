import datetime
import time
from functools import wraps


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution of {func.__name__} took {end_time - start_time} seconds")
        return result

    return wrapper


def generate_runid():
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{datetime_str}"
