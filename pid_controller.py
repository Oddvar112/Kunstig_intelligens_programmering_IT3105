import jax.numpy as jnp

from controller import Controller

class PIDController(Controller):
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.01):
        self.params = jnp.array([kp, ki, kd], dtype=jnp.float32)  
        
    def reset(self):
        pass  
    
    def get_params(self):
        return self.params
    
    def set_params(self, params):
        self.params = params
    
    def update_params(self, params, grads, learning_rate):
        return params - learning_rate * grads
    
    def get_param_history_entry(self, params):
        return {'kp': float(params[0]), 'ki': float(params[1]), 'kd': float(params[2])}
    
    def compute_control(self, error_history):
        
        kp, ki, kd = self.params
        
        #error siste
        E = error_history[-1]
        
        # Integral
        E_integral = jnp.sum(jnp.array(error_history))
        
        # deriverte 
        if len(error_history) > 1:
            E_derivative = error_history[-1] - error_history[-2]
        else:
            E_derivative = 0.0
        
        # PID formel
        U = kp * E + ki * E_integral + kd * E_derivative
        return U