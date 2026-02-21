import jax
import jax.numpy as jnp
from Prosjekt_1.plant import Plant

def soft_clip(x, low=0.0, high=1.0):
    return low + (high - low) * jax.nn.sigmoid(4.0 * (x - (low + high) / 2)) #prøvde med hard clippinng men dennne er bedre da vvi alltidd får en gradient 

class CournotPlant(Plant):
    def __init__(self, pmax=2.0, cm=0.1, target_profit=0.5, q1_init=0.5, q2_init=0.5):
        self.pmax = pmax
        self.cm = cm
        self.target_profit = target_profit
        self.q1_init = q1_init
        self.q2_init = q2_init
    
    def reset(self):
        return jnp.array([self.q1_init, self.q2_init])
    
    def step(self, u, noise, state):
        q1, q2 = state

        # ∂MSE/∂params = ∂MSE/∂error × ∂error/∂profit × ∂profit/∂q1_new × ∂q1_new/∂u × ∂u/∂params
        
        q1_new = soft_clip(q1 + u, 0.0, 1.0)
        q2_new = soft_clip(q2 + noise, 0.0, 1.0)

        q_total = q1_new + q2_new
        price = jnp.maximum(self.pmax - q_total, 0.0)
        profit = q1_new * (price - self.cm)
        
        error = self.target_profit - profit
        
        new_state = jnp.array([q1_new, q2_new])
        
        return profit, error, new_state
    
    def get_target(self):        
        return self.target_profit