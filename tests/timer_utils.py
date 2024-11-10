import time

def timer(func):
    """Decorator to time a function's execution."""
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Executime time: ", end - start, " seconds")
        return res
    return wrapper