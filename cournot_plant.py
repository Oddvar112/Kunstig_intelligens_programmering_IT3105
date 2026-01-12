import jax.numpy as jnp
from plant import Plant

class CournotPlant(Plant):
    def __init__(self, pmax=2.0, cm=0.1, target_profit=0.5, q1_init=0.5, q2_init=0.5):
        self.pmax = pmax
        self.cm = cm
        self.target_profit = target_profit
        self.q1_init = q1_init
        self.q2_init = q2_init
    
    def reset(self):
        return jnp.array([self.q1_init, self.q2_init])
    
    def step(self, u, d, state):
        q1, q2 = state
        
        q1_new = jnp.clip(q1 + u, 0.0, 1.0)
        q2_new = jnp.clip(q2 + d, 0.0, 1.0)
        
        q_total = q1_new + q2_new
        price = jnp.maximum(self.pmax - q_total, 0.0)
        profit = q1_new * (price - self.cm)
        
        error = self.target_profit - profit
        
        new_state = jnp.array([q1_new, q2_new])
        
        return profit, error, new_state
    
    def get_target(self):        
        return self.target_profit