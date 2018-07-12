### CLASS Memory
from collections import deque
import numpy as np

class Memory:
    def __init__ (self, max_size=1000):
        self.__buffer = deque (maxlen=max_size)
    
    def add (self, sample):
        self.__buffer.append (sample)
            
    def sample (self, batch_size):
        idx = np.random.choice (np.arange (len (self.__buffer)), size=batch_size, replace=False)
        return [self.__buffer[ii] for ii in idx]
    
    def length (self):
        return len (self.__buffer)