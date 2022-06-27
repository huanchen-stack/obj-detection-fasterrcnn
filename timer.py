from sigfig import round
import time

class Clock(object):

    def __init__(self):
        super().__init__()
        self.agg = 0
        self.start = None
        self.diff = None

    def tic(self):
        self.start = time.time()
    
    def toc(self):
        self.diff = time.time() - self.start
        self.agg += self.diff
    
    def get_time(self):
        return round(self.diff, sigfigs=4)

    def get_agg(self):
        return round(self.agg, sigfigs=4)