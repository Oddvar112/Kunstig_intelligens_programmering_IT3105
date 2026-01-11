import jax
import jax.numpy as jnp

class NeuralPIDController:
    def __init__(self, layers=[3, 8, 8, 1], activation='tanh'):
        self.layers = layers
        if activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'sigmoid':
            self.activation = jax.nn.sigmoid
        elif activation == 'relu':
            self.activation = jax.nn.relu
        else:
            self.activation = lambda x: x
        
        self.weight_matrices = []
        self.bias_vectors = []
        
        key = jax.random.PRNGKey(0)
        
        for fan_in, fan_out in zip(layers[:-1], layers[1:]):
            key, wkey = jax.random.split(key)
            
            W = jax.random.uniform(
                wkey,
                (fan_in, fan_out),
                minval=-0.5,
                maxval=0.5
            )
            
            b = jnp.zeros(fan_out)
            
            self.weight_matrices.append(W)
            self.bias_vectors.append(b)
    
    def network_forward(self, x, weights, biases):
        a = x
        for i, (W, b) in enumerate(zip(weights, biases)):
            a = a @ W + b
            if i < len(weights) - 1:
                a = self.activation(a)
        
        return a
    
    def get_params(self):
        params = []
        for W, b in zip(self.weight_matrices, self.bias_vectors):
            params.append({'w': W, 'b': b})
        return params
    
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
        
        U = output[0] if output.shape == (1,) else output
        
        return U