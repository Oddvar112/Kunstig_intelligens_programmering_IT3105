import jax
import jax.numpy as jnp

#https://www.youtube.com/watch?v=Oieh4YFZZz0

class NeuralPIDController:
    def __init__(self, layers=[3, 8, 8, 1], activation='tanh', weight_init_range=(-0.5, 0.5), bias_init_range=(0.0, 0.0)):
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