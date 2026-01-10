import jax.numpy as jnp
from plant import Plant

class BathtubPlant(Plant):
    def __init__(self, A=1.0, C=0.01, H0=1.0):
        self.A = A
        self.C = C
        self.H0 = H0
        self.g = 9.8
    
    def reset(self):
        return self.H0
    
    def step(self, u, d, state):

        # formeler ifra oppgave 
        V = jnp.sqrt(2 * self.g * jnp.maximum(state, 0))
        Q = V * self.C
        dB = u + d - Q
        dH = dB / self.A
        new_state = jnp.maximum(state + dH, 0)
        
        output = new_state
        error = self.H0 - new_state
        
        return output, error, new_state
    
    def get_target(self):
        return self.H0