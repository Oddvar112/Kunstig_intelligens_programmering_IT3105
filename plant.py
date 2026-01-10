from abc import ABC, abstractmethod

class Plant(ABC):
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, u, d):
        pass
    
    @abstractmethod
    def get_target(self):
        pass