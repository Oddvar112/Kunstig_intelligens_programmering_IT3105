from abc import ABC, abstractmethod

class Controller(ABC):
    
    @abstractmethod
    def get_params(self):
        pass
    
    @abstractmethod
    def set_params(self, params):
        pass
    
    @abstractmethod
    def compute_control(self, error_history):
        pass
    
    @abstractmethod
    def update_params(self, params, grads, learning_rate):
        pass
    
    @abstractmethod
    def get_param_history_entry(self, params):
        pass