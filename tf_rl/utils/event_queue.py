import time

from queue import PriorityQueue

class EqItem(object):
    """Function and sechduled execution timestamp.

    This class is needed because if
    we use tuple instead, Python will ocassionally
    complaint that it does not know how to compare
    functions"""
    def __init__(self, ts, f):
        self.ts = ts
        self.f  = f
        
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
    
    def schedule(self, f, ts):
        """Schedule f to be execute at time ts"""
        self.q.put(EqItem(ts, f))
        
    def schedule_recurring(self, f, interval):
        """Schedule f to be run every interval seconds.

	It will be run for the first time interval seconds
        from now"""
        def recuring_f():
            f()
            self.schedule(recuring_f, time.time() + interval)
        self.schedule(recuring_f, time.time() + interval)
        
        
    def run(self):
        """Execute events in the queue as timely as possible."""
        while True:
            event = self.q.get()
            now = time.time()
            if now < event.ts:
                time.sleep(event.ts - now)
            event.f()
            
