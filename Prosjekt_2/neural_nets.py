import jax
import jax.numpy as jnp


def init_network_params(layer_sizes, key, weight_range=(-0.3, 0.3)):
    params = []
    for i in range(len(layer_sizes) - 1):
        dim_in = layer_sizes[i]
        dim_out = layer_sizes[i + 1]

        key, wkey, bkey = jax.random.split(key, 3)

        # Xavier initialisering
        scale = jnp.sqrt(2.0 / (dim_in + dim_out))
        W = jax.random.normal(wkey, (dim_in, dim_out)) * scale
        b = jnp.zeros(dim_out)

        params.append({'w': W, 'b': b})
    return params


def forward(params, x, activations):
    a = x
    for i, layer in enumerate(params):
        a = a @ layer['w'] + layer['b']
        if i < len(params) - 1:  # aktivering på alle lag unntatt siste
            if i < len(activations):
                act = activations[i]
                if act == 'relu':
                    a = jax.nn.relu(a)
                elif act == 'tanh':
                    a = jnp.tanh(a)
                elif act == 'sigmoid':
                    a = jax.nn.sigmoid(a)
    return a


def nnr_forward(params, game_states_flat, activations):
    output = forward(params, game_states_flat, activations)
    return jnp.tanh(output)


def nnd_forward(params, abstract_state, action_onehot, activations, abstract_state_size):
    x = jnp.concatenate([abstract_state, action_onehot])
    output = forward(params, x, activations)

    # Split output: abstract_state_size elementer for state, 1 for reward
    next_state = output[:abstract_state_size]
    reward = output[abstract_state_size]

    # Normaliser next_state for å unngå at verdier eksploderer over tid
    # Bruker tanh for å holde verdier i [-1, 1]
    next_state = jnp.tanh(next_state)

    return next_state, reward


def nnp_forward(params, abstract_state, activations, num_actions):
    output = forward(params, abstract_state, activations)
    # Split: num_actions elementer for policy logits, 1 for value
    policy_logits = output[:num_actions]
    value = output[num_actions]
    # Softmax for policy-distribusjon
    policy = jax.nn.softmax(policy_logits)

    return policy, value
