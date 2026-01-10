import jax
import jax.numpy as jnp
from jax import grad
import numpy as np

class CONSYS:
    
    def __init__(self, controller, plant, config):
        self.controller = controller
        self.plant = plant
        self.config = config
    
    def run_epoch(self, params, noise):
        self.controller.set_params(params)
        self.controller.reset()
        
        # Få initial state fra plant
        state = self.plant.reset()
        
        error_history = []
        
        for t in range(self.config.num_timesteps):
            if len(error_history) == 0:
                error_history.append(0.0)
            
            u = self.controller.compute_control(error_history)
            d = noise[t]
            
            # JAX kan trace med state sendt inn
            output, error, state = self.plant.step(u, d, state)
            error_history.append(error)
        
        errors = jnp.array(error_history[1:])
        mse = jnp.mean(errors ** 2)
        
        return mse
    
    def loss_function(self, params, noise):
        mse = self.run_epoch(params, noise)
        return mse
    
    def train(self):
        params = self.controller.get_params()
        
        print(f"Initial params: kp={params[0]:.4f}, ki={params[1]:.4f}, kd={params[2]:.4f}")
        
        mse_history = []
        kp_history = []
        ki_history = []
        kd_history = []
        
        grad_fn = jax.grad(self.loss_function)
        
        for epoch in range(self.config.num_epochs):
            noise = np.random.uniform(self.config.noise_range[0],self.config.noise_range[1],size=self.config.num_timesteps)
            
            mse = self.loss_function(params, noise)
            grads = grad_fn(params, noise)
            
           
            print(f"Epoch {epoch}: MSE={mse:.6f}, grads={grads}")
            
            params = params - self.config.learning_rate * grads
            
            mse_history.append(float(mse))
            kp_history.append(float(params[0]))
            ki_history.append(float(params[1]))
            kd_history.append(float(params[2]))
        
        print(f"Final params: kp={params[0]:.4f}, ki={params[1]:.4f}, kd={params[2]:.4f}")
        
        self.controller.set_params(params)
        
        results = {
            'mse_history': mse_history,
            'kp_history': kp_history,
            'ki_history': ki_history,
            'kd_history': kd_history
        }
        
        return results