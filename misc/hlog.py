from contextlib import contextmanager
import logging
import threading
import time

state = threading.local()
state.path = []

@contextmanager
def task(name, log_time=True):
    state.path.append(name)
    begin = time.time()
    yield
    end = time.time()
    if log_time:
        log("done in %0.2f" % (end - begin))
    state.path.pop()

def log(value):
    if isinstance(value, float):
        value = "%0.4f" % value
    print('%s %s' % ('/'.join(state.path), value))

def value(name, value):
    with task(name, log_time=False):
        log(value)

def loop(template, coll, log_time=True):
    for i, item in enumerate(coll):
        with task(template % i, log_time):
            yield item

def fn(name, log_time=True):
    def wrap(underlying):
        def wrapped(*args, **kwargs):
            with task(name, log_time):
                underlying(*args, **kwargs)
        return wrapped
    return wrap
