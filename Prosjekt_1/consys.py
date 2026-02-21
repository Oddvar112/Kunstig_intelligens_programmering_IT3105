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
            noise = np.random.uniform(
                self.config.noise_range[0],
                self.config.noise_range[1],
                size=self.config.num_timesteps
            )
            
            mse = self.loss_function(params, noise)
            grads = grad_fn(params, noise)
            
            """          
            For NN:
            params = [
                {'w': W1 (3×8), 'b': b1 (8,)},   # Vektene i lag 1 
                {'w': W2 (8×8), 'b': b2 (8,)},   # Vektene i lag 2
                {'w': W3 (8×1), 'b': b3 (1,)}    # Vektene i lag 3
            ]

            grads = [
                {'w': ∂MSE/∂W1 (3×8), 'b': ∂MSE/∂b1 (8,)},  # Deriverte for lag 1
                {'w': ∂MSE/∂W2 (8×8), 'b': ∂MSE/∂b2 (8,)},  # Deriverte for lag 2
                {'w': ∂MSE/∂W3 (8×1), 'b': ∂MSE/∂b3 (1,)}   # Deriverte for lag 3
            ]
            
            W1 = [
                [w₁₁, w₁₂, w₁₃, w₁₄, w₁₅, w₁₆, w₁₇, w₁₈],  # 8 vekter
                [w₂₁, w₂₂, w₂₃, w₂₄, w₂₅, w₂₆, w₂₇, w₂₈],  # 8 vekter
                [w₃₁, w₃₂, w₃₃, w₃₄, w₃₅, w₃₆, w₃₇, w₃₈]   # 8 vekter
            ]

            # ∂MSE/∂W1 er også en (3×8) matrise med 24 gradienter:
            ∂MSE/∂W1 = [
                [∂MSE/∂w₁₁, ∂MSE/∂w₁₂, ∂MSE/∂w₁₃, ..., ∂MSE/∂w₁₈],  # 8 gradienter
                [∂MSE/∂w₂₁, ∂MSE/∂w₂₂, ∂MSE/∂w₂₃, ..., ∂MSE/∂w₂₈],  # 8 gradienter
                [∂MSE/∂w₃₁, ∂MSE/∂w₃₂, ∂MSE/∂w₃₃, ..., ∂MSE/∂w₃₈]   # 8 gradienter
            ]
            
            For PID:
            params = [kp, ki, kd]  # 3 verdier
            grads = [∂MSE/∂kp, ∂MSE/∂ki, ∂MSE/∂kd]  # 3 gradienter
            """
            
            print(f"Epoch {epoch}: MSE={mse:.6f}")
            
            params = self.controller.update_params(params, grads, self.config.learning_rate)
            
            # Lagre PID-historikk hvis det er en PID-controller (3 parametre)
            param_entry = self.controller.get_param_history_entry(params)
            if param_entry is not None:
                kp_history.append(param_entry['kp'])
                ki_history.append(param_entry['ki'])
                kd_history.append(param_entry['kd'])
            
            mse_history.append(float(mse))
        
        self.controller.set_params(params)
        
        results = {'mse_history': mse_history}
        if kp_history:
            results['kp_history'] = kp_history
            results['ki_history'] = ki_history
            results['kd_history'] = kd_history
        
        return results