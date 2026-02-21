import jax
import jax.numpy as jnp

from Prosjekt_1.controller import Controller

#https://www.youtube.com/watch?v=Oieh4YFZZz0 used for inspiration and help we could not use the same setup :(

class NeuralPIDController(Controller):
    def __init__(self, layers=[3, 8, 8, 1], activations=['tanh', 'tanh'], weight_init_range=(-0.5, 0.5), bias_init_range=(0.0, 0.0)):
        self.activations = []
        for act in activations:
            if act == 'tanh':
                self.activations.append(jnp.tanh)
            elif act == 'sigmoid':
                self.activations.append(jax.nn.sigmoid)
            elif act == 'relu':
                self.activations.append(jax.nn.relu)
            else:
                self.activations.append(lambda x: x) 
        
        self.weight_matrices = []
        self.bias_vectors = []
        
        key = jax.random.PRNGKey(0)
        
        for i in range(len(layers) - 1):
            dimInput = layers[i]
            dimOutput = layers[i + 1]
            
            key, wkey, bkey = jax.random.split(key, 3)
            W = jax.random.uniform(wkey, (dimInput, dimOutput), minval=weight_init_range[0], maxval=weight_init_range[1])
            if bias_init_range[0] == bias_init_range[1]:
                b = jnp.full(dimOutput, bias_init_range[0])
            else:
                b = jax.random.uniform(bkey, (dimOutput,), minval=bias_init_range[0], maxval=bias_init_range[1])
            
            self.weight_matrices.append(W)
            self.bias_vectors.append(b)
    
    def network_forward(self, x, weights, biases):
        a = x
        for i, (W, b) in enumerate(zip(weights, biases)):
            a = a @ W + b
            if i < len(weights) - 1:  
                a = self.activations[i](a) # riktig activation for laget 
        return a
    
    def get_param_history_entry(self, params):
        return None
    
    def get_params(self):
        params = []
        for W, b in zip(self.weight_matrices, self.bias_vectors):
            params.append({'w': W, 'b': b})
        return params
    
    def update_params(self, params, grads, learning_rate):
        new_params = []
        for layer_params, layer_grads in zip(params, grads):
            new_layer = {
                'w': layer_params['w'] - learning_rate * layer_grads['w'],
                'b': layer_params['b'] - learning_rate * layer_grads['b']
            }
            new_params.append(new_layer)
        return new_params
    
    def set_params(self, params):
        self.weight_matrices = [p['w'] for p in params]
        self.bias_vectors = [p['b'] for p in params]
    
    def compute_control(self, error_history):
        E = error_history[-1]
        E_integral = jnp.sum(jnp.array(error_history))
        
        if len(error_history) > 1:
            E_derivative = error_history[-1] - error_history[-2]
        else:
            E_derivative = 0.0
        
        x = jnp.array([E, E_derivative, E_integral])
        
        output = self.network_forward(x, self.weight_matrices, self.bias_vectors)
        
        U = output[0]
        
        return U