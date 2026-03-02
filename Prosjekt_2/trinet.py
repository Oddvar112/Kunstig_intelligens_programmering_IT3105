"""
TrinetManager - Manager for MuZeros tre nevrale nettverk (Psi).

Psi består av:
- NNr (Representation): game states -> abstract state
- NNd (Dynamics): (abstract state, action) -> (next state, reward)
- NNp (Prediction): abstract state -> (policy, value)

Håndterer initialisering, inference og BPTT-trening med Adam optimizer.
JIT-kompilert for ytelse.
"""

import jax
import jax.numpy as jnp
import numpy as np

from neural_nets import init_network_params, nnr_forward, nnd_forward, nnp_forward


class TrinetManager:
    """Manager for MuZeros tre nettverk (Psi)."""

    def __init__(self, config):
        self.config = config
        
        # Adam optimizer state
        self.adam_state = None
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_eps = 1e-8
        self.adam_t = 0

        # ============ JIT-kompilerte funksjoner ============
        # Disse kjører mye raskere enn vanlig Python fordi JAX
        # kompilerer dem til optimalisert maskinkode ved første kall.

        # Inference-funksjoner (brukes tusenvis av ganger i MCTS)
        nnr_acts = config.nnr_activations
        nnd_acts = config.nnd_activations
        nnp_acts = config.nnp_activations
        abs_size = config.abstract_state_size
        n_actions = config.num_actions

        self._jitted_nnr = jax.jit(
            lambda params, x: nnr_forward(params, x, nnr_acts)
        )
        self._jitted_nnd = jax.jit(
            lambda params, s, a: nnd_forward(params, s, a, nnd_acts, abs_size)
        )
        self._jitted_nnp = jax.jit(
            lambda params, s: nnp_forward(params, s, nnp_acts, n_actions)
        )

        # Trenings-funksjon (loss + gradienter i ett kall)
        self._jitted_loss_and_grad = jax.jit(
            jax.value_and_grad(self._bptt_loss)
        )

    def init_params(self, key):
        """Initialiser parametre for alle tre nettverk."""
        key1, key2, key3 = jax.random.split(key, 3)

        return {
            'nnr': init_network_params(self.config.nnr_layers, key1),
            'nnd': init_network_params(self.config.nnd_layers, key2),
            'nnp': init_network_params(self.config.nnp_layers, key3)
        }

    def representation(self, psi_params, game_states_flat):
        """NNr: Konverter game states til abstract state. (JIT)"""
        return self._jitted_nnr(psi_params['nnr'], game_states_flat)

    def dynamics(self, psi_params, abstract_state, action):
        """NNd: Prediker neste abstract state og reward. (JIT)"""
        action_onehot = jax.nn.one_hot(action, self.config.num_actions)
        return self._jitted_nnd(psi_params['nnd'], abstract_state, action_onehot)

    def prediction(self, psi_params, abstract_state):
        """NNp: Prediker policy og value fra abstract state. (JIT)"""
        return self._jitted_nnp(psi_params['nnp'], abstract_state)

    def train_step(self, psi_params, episode_buffer):
        """Utfør ett treningssteg med BPTT og Adam optimizer. (JIT)"""
        minibatch = episode_buffer.sample_minibatch()
        if minibatch is None:
            return psi_params, 0.0

        total_loss = 0.0
        total_grads = None

        # Akkumuler gradienter over minibatch
        for sample in minibatch:
            states_flat, actions, target_policies, target_values, target_rewards = sample

            states_flat = jnp.array(states_flat)
            actions_onehot = jnp.array([
                jax.nn.one_hot(a, self.config.num_actions) for a in actions
            ])
            target_policies = jnp.array(target_policies)
            target_values = jnp.array(target_values)
            target_rewards = jnp.array(target_rewards)

            # Bruker JIT-kompilert loss+grad
            loss, grads = self._jitted_loss_and_grad(
                psi_params,
                states_flat,
                actions_onehot,
                target_policies,
                target_values,
                target_rewards
            )

            total_loss += float(loss)

            if total_grads is None:
                total_grads = grads
            else:
                total_grads = jax.tree.map(lambda g1, g2: g1 + g2, total_grads, grads)

        # Gjennomsnittlig gradient
        n = len(minibatch)
        avg_grads = jax.tree.map(lambda g: g / n, total_grads)
        avg_loss = total_loss / n

        # Gradient clipping (max norm = 5.0)
        grad_norm = jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree.leaves(avg_grads)
        ))
        max_norm = 5.0
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
        avg_grads = jax.tree.map(lambda g: g * clip_factor, avg_grads)

        # Adam update
        updated_params = self._adam_update(psi_params, avg_grads)

        # Lagre diagnostikk
        self.last_diagnostics = {
            'grad_norm': float(grad_norm),
            'grad_norm_clipped': float(grad_norm * clip_factor),
            'clip_factor': float(clip_factor),
            'loss': avg_loss
        }

        return updated_params, avg_loss

    def _adam_update(self, params, grads):
        """Oppdater parametre med Adam optimizer."""
        self.adam_t += 1

        if self.adam_state is None:
            self.adam_state = {
                'm': jax.tree.map(jnp.zeros_like, params),
                'v': jax.tree.map(jnp.zeros_like, params)
            }

        # Oppdater moment-estimater
        self.adam_state['m'] = jax.tree.map(
            lambda m, g: self.adam_beta1 * m + (1 - self.adam_beta1) * g,
            self.adam_state['m'], grads
        )
        self.adam_state['v'] = jax.tree.map(
            lambda v, g: self.adam_beta2 * v + (1 - self.adam_beta2) * g ** 2,
            self.adam_state['v'], grads
        )

        # Bias-korreksjon
        bc1 = 1 - self.adam_beta1 ** self.adam_t
        bc2 = 1 - self.adam_beta2 ** self.adam_t

        # Oppdater parametre
        return jax.tree.map(
            lambda p, m, v: p - self.config.learning_rate * (m / bc1) / (jnp.sqrt(v / bc2) + self.adam_eps),
            params, self.adam_state['m'], self.adam_state['v']
        )

    def _bptt_loss(self, psi_params, states_flat, actions_onehot, target_policies, target_values, target_rewards):
        """Beregn BPTT loss for alle tre nettverk.
        MÅ være ren JAX-funksjon for jax.grad og jax.jit.
        """
        # Representasjon: game states -> abstract state
        sigma = nnr_forward(
            psi_params['nnr'],
            states_flat,
            self.config.nnr_activations
        )

        # Prediksjon for initial state
        pred_policy, pred_value = nnp_forward(
            psi_params['nnp'],
            sigma,
            self.config.nnp_activations,
            self.config.num_actions
        )

        # Policy loss (cross-entropy)
        loss = -jnp.sum(target_policies[0] * jnp.log(pred_policy + 1e-8))
        
        # Entropy regularisering
        entropy = -jnp.sum(pred_policy * jnp.log(pred_policy + 1e-8))
        loss -= 0.05 * entropy
        
        # Value loss (MSE, økt vekt)
        loss += 2.0 * jnp.square(pred_value - target_values[0])

        # Roll-ahead med dynamics network
        w = actions_onehot.shape[0]
        for j in range(w):
            sigma, pred_reward = nnd_forward(
                psi_params['nnd'],
                sigma,
                actions_onehot[j],
                self.config.nnd_activations,
                self.config.abstract_state_size
            )

            pred_policy, pred_value = nnp_forward(
                psi_params['nnp'],
                sigma,
                self.config.nnp_activations,
                self.config.num_actions
            )

            # Akkumuler loss
            loss += -jnp.sum(target_policies[j + 1] * jnp.log(pred_policy + 1e-8))
            entropy = -jnp.sum(pred_policy * jnp.log(pred_policy + 1e-8))
            loss -= 0.05 * entropy
            loss += 2.0 * jnp.square(pred_value - target_values[j + 1])
            loss += jnp.square(pred_reward - target_rewards[j])

        return loss / (w + 1)


    def compute_diagnostics(self, psi_params, episode_buffer):
        """Beregn detaljert diagnostikk for debugging."""
        minibatch = episode_buffer.sample_minibatch()
        if minibatch is None:
            return None

        diagnostics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'reward_loss': 0.0,
            'entropy': 0.0,
            'pred_values': [],
            'target_values': [],
            'pred_policies': [],
            'target_policies': [],
            'pred_rewards': [],
            'target_rewards': [],
            'abstract_states': []
        }

        for sample in minibatch:
            states_flat, actions, target_policies, target_values, target_rewards = sample

            # Forward pass (bruker JIT-versjoner)
            states_flat = jnp.array(states_flat)
            sigma = self._jitted_nnr(psi_params['nnr'], states_flat)
            diagnostics['abstract_states'].append(np.array(sigma))

            pred_policy, pred_value = self._jitted_nnp(psi_params['nnp'], sigma)

            # Samle predictions
            diagnostics['pred_values'].append(float(pred_value))
            diagnostics['target_values'].append(float(target_values[0]))
            diagnostics['pred_policies'].append(np.array(pred_policy))
            diagnostics['target_policies'].append(np.array(target_policies[0]))

            # Beregn individuelle losses
            policy_loss = -np.sum(target_policies[0] * np.log(np.array(pred_policy) + 1e-8))
            entropy = -np.sum(np.array(pred_policy) * np.log(np.array(pred_policy) + 1e-8))
            value_loss = 0.5 * (float(pred_value) - float(target_values[0])) ** 2

            diagnostics['policy_loss'] += policy_loss
            diagnostics['value_loss'] += value_loss
            diagnostics['entropy'] += entropy

            # Rollout rewards (bruker JIT-versjoner)
            for j, a in enumerate(actions):
                action_onehot = jax.nn.one_hot(a, self.config.num_actions)
                sigma, pred_reward = self._jitted_nnd(psi_params['nnd'], sigma, action_onehot)
                diagnostics['pred_rewards'].append(float(pred_reward))
                diagnostics['target_rewards'].append(float(target_rewards[j]))
                diagnostics['reward_loss'] += (float(pred_reward) - float(target_rewards[j])) ** 2

        # Normaliser
        n = len(minibatch)
        diagnostics['policy_loss'] /= n
        diagnostics['value_loss'] /= n
        diagnostics['reward_loss'] /= n
        diagnostics['entropy'] /= n

        # Abstract state statistikk
        abstract_states = np.array(diagnostics['abstract_states'])
        diagnostics['abstract_state_mean'] = np.mean(abstract_states)
        diagnostics['abstract_state_std'] = np.std(abstract_states)
        diagnostics['abstract_state_min'] = np.min(abstract_states)
        diagnostics['abstract_state_max'] = np.max(abstract_states)

        # Sjekk for representasjonskollaps
        if len(abstract_states) > 1:
            pairwise_distances = []
            for i in range(min(10, len(abstract_states))):
                for j in range(i + 1, min(10, len(abstract_states))):
                    dist = np.linalg.norm(abstract_states[i] - abstract_states[j])
                    pairwise_distances.append(dist)
            diagnostics['abstract_state_diversity'] = np.mean(pairwise_distances) if pairwise_distances else 0.0
        else:
            diagnostics['abstract_state_diversity'] = 0.0

        return diagnostics