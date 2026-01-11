import jax
import jax.numpy as jnp
from jax import grad
import numpy as np
from nn_pid_controller import NeuralPIDController

class CONSYS:
    
    def __init__(self, controller, plant, config):
        self.controller = controller
        self.plant = plant
        self.config = config
        self.is_neural = isinstance(controller, NeuralPIDController)
    
    def run_epoch(self, params, noise):
        self.controller.set_params(params)
        state = self.plant.reset()
        error_history = []
        
        for t in range(self.config.num_timesteps):
            if len(error_history) == 0:
                error_history.append(0.0)
            
            u = self.controller.compute_control(error_history)
            d = noise[t]
            output, error, state = self.plant.step(u, d, state)
            error_history.append(error)
        
        errors = jnp.array(error_history[1:])
        mse = jnp.mean(errors ** 2)
        return mse
    
    def loss_function(self, params, noise):
        return self.run_epoch(params, noise)
    
    def train(self):
        params = self.controller.get_params()
        mse_history = []
        kp_history = []
        ki_history = []
        kd_history = []
        
        grad_fn = jax.grad(self.loss_function)
        
        for epoch in range(self.config.num_epochs):
            noise = np.random.uniform(self.config.noise_range[0],self.config.noise_range[1],size=self.config.num_timesteps)
            
            mse = self.loss_function(params, noise)
            grads = grad_fn(params, noise)
            
            # Oppdater basert på type
            if self.is_neural:
                # NN
                print(f"Epoch {epoch}: MSE={mse:.6f}")
                new_params = []
                for i in range(len(params)):
                    layer_params = params[i]
                    layer_grads = grads[i]
                    new_layer = {
                        'w': layer_params['w'] - self.config.learning_rate * layer_grads['w'],
                        'b': layer_params['b'] - self.config.learning_rate * layer_grads['b']
                    }
                    new_params.append(new_layer)
                params = new_params
            else:
                # normal
                print(f"Epoch {epoch}: MSE={mse:.6f}, grads={grads}")

                params = params - self.config.learning_rate * grads
                kp_history.append(float(params[0]))
                ki_history.append(float(params[1]))
                kd_history.append(float(params[2]))
            
            mse_history.append(float(mse))
        
        self.controller.set_params(params)
        
        results = {'mse_history': mse_history}
        if not self.is_neural:
            results['kp_history'] = kp_history
            results['ki_history'] = ki_history
            results['kd_history'] = kd_history
        
        return results