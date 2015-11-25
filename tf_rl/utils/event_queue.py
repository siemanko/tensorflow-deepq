import time

from collections import defaultdict
from queue import PriorityQueue

class EqItem(object):
    """Function and sechduled execution timestamp.

    This class is needed because if
    we use tuple instead, Python will ocassionally
    complaint that it does not know how to compare
    functions"""
    def __init__(self, ts, f, can_skip=False, recurrence_interval=None, name=None):
        self.ts                  = ts
        self.f                   = f
        self.can_skip            = can_skip
        self.recurrence_interval = recurrence_interval
        self.name                = name

    def __lt__(self, other):
        return self.ts < other.ts

    def __eq__(self, other):
        return self.ts == other.ts

class EventQueue(object):
    def __init__(self):
        """Event queue for executing events at
        specific timepoints.

        In current form it is NOT thread safe."""
        self.q = PriorityQueue()

    def schedule(self, f, ts=None, can_skip=False, recurrence_interval=None, name=None):
        """Schedule f to be execute at time ts"""
        ts = ts or time.time()
        self.q.put(EqItem(ts, f, can_skip=can_skip, recurrence_interval=recurrence_interval, name=name))

    def run(self, run_for=None):
        """Execute events in the queue as timely as possible."""
        self.stats = defaultdict(lambda: 0)
        self.running_since = time.time()
        self.running_until = None
        self.spare_time    = 0
        try:
            while run_for is None or (time.time() - self.running_since) < run_for:
                event = self.q.get()
                now = time.time()
                arrived_on_time = False
                if now < event.ts:
                    self.spare_time += (event.ts - now)
                    time.sleep(event.ts - now)
                    arrived_on_time = True
                if arrived_on_time or not event.can_skip:
                    if event.name is not None:
                        self.stats[event.name] += 1
                    event.f()
                if event.recurrence_interval is not None:
                    event.ts += event.recurrence_interval
                    self.q.put(event)
        finally:
            self.running_until = time.time()

    def statistics_str(self):
        total_time    = (self.running_until or time.time()) - self.running_since
        executions = ', '.join(['%s: %.1f / s' % (name, float(occurences) / total_time)
                          for name, occurences in self.stats.items()])
        spare_time = '(spare time: %.1f %% )' % (100.0 * self.spare_time / total_time)
        return [executions, spare_time]
