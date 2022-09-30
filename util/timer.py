# from sigfig import round
import time
import torch

class Clock(object):
    
    def __init__(self):
        super().__init__()
        self.agg = 0
        self.start = None
        self.diff = None
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
    
    def tic(self):
        torch.cuda.synchronize()
        self.starter.record()
    
    def toc(self):
        self.ender.record()
        torch.cuda.synchronize()
    
    def get_time(self):
        delta = self.starter.elapsed_time(self.ender)/1000
        self.agg += delta
        return round(delta, 6)

    def get_agg(self):
        return round(self.agg, 6)
