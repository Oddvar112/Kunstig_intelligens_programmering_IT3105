import jax
import jax.numpy as jnp
from plant import Plant

'''
d = demand shock
u = interest rate change
r = interest rate
pi = inflation
target_pi = inflation target
alpha = sensitivity of r on pi

'''


class inflation_plant(Plant):
    def __init__(
            self,
            r0=0.04, # Current interest rate
            pi0=0.031, # inflation 2025
            target_pi=0.02, # Norges bank target
            alpha=0.5):
        
        self.pi0 = pi0 # starting inflation
        self.r0 = r0 # starting interesr rate
        self.target_pi = target_pi # goal
        self.alpha = alpha

        self.reset()
            

    def reset(self):
        #sets inflation and rate to default values
        self.pi=self.pi0
        self.r=self.r0
        return (self.pi, self.r)


    def step(self, u, d, state=None):
        self.r = self.r + u # Change in interest rate
        self.pi = self.pi - self.alpha * (self.r) + d # Change in inflation due to inflation shock

        output = self.pi
        error = self.target_pi - self.pi
        new_state = (self.pi, self.r)
        return output, error, new_state
    
    def get_target(self):
        # Returns inflation target
        return self.target_pi

