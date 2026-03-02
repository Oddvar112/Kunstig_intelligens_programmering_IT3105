"""
Monte Carlo Tree Search (MCTS) for MuZero.

Implementerer u-MCTS som bruker nevrale nettverk for:
- Dynamics: predikere neste state og reward
- Prediction: predikere policy og value

Bruker PUCT (Polynomial Upper Confidence Trees) for tree policy.
"""

import math

import numpy as np
import jax.numpy as jnp


class MCTSNode:
    """Node i MCTS-treet.

    Attributter:
        abstract_state: Abstract state-vektor fra representation network
        parent: Forelder-node
        action: Handling som førte til dette nodet
        reward: Belønning fra å ta handlingen
        prior: Prior probability fra policy network
        children: Dict med action -> child node
        visit_count: Antall besøk
        total_value: Sum av verdier fra backpropagation
    """

    def __init__(self, abstract_state, parent=None, action=None, reward=0.0, prior=0.0):
        self.abstract_state = abstract_state
        self.parent = parent
        self.action = action
        self.reward = reward
        self.prior = prior

        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False

    @property
    def q_value(self):
        """Gjennomsnittlig verdi for dette nodet."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class UMCTS:
    """MuZero Monte Carlo Tree Search.

    Bruker abstract states fra nevrale nettverk for søk.
    """

    def __init__(self, config, trinet_manager):
        self.config = config
        self.trinet = trinet_manager
        self.num_actions = config.num_actions
        self.last_root = None

    def search(self, psi_params, root_abstract_state, legal_actions):
        """Utfør MCTS-søk og returner policy og value.

        Args:
            psi_params: nettverksparametre
            root_abstract_state: abstract state for nåværende posisjon
            legal_actions: liste med lovlige handlinger

        Returns:
            (policy, value) - policy er sannsynlighetsdistribusjon over handlinger
        """
        root = MCTSNode(abstract_state=root_abstract_state)

        # Hent initial policy og value fra prediction network
        init_policy, init_value = self.trinet.prediction(psi_params, root_abstract_state)
        init_policy = np.array(init_policy)

        # Legg til Dirichlet-støy på root for utforsking
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * self.num_actions)
        eps = self.config.dirichlet_epsilon
        noisy_policy = (1 - eps) * init_policy + eps * noise

        # Expand root med støyete policy
        self._expand(root, psi_params, noisy_policy)

        # Tracking av min/max Q-verdier for normalisering
        min_q = float('inf')
        max_q = float('-inf')

        # Utfør Ms søk
        for m in range(self.config.num_mcts_searches):
            # 1. Tree policy: traverser til leaf
            leaf, depth = self._tree_policy(root, min_q, max_q)

            # 2. Expand leaf om nødvendig
            if not leaf.is_expanded:
                self._expand(leaf, psi_params)

            # 3. Velg tilfeldig child for rollout
            if leaf.children:
                random_action = np.random.choice(list(leaf.children.keys()))
                child = leaf.children[random_action]

                # 4. Rollout fra child
                remaining_depth = max(0, self.config.max_search_depth - depth - 1)
                accum_rewards = self._rollout(child, remaining_depth, psi_params)

                # 5. Backpropagation
                self._backpropagate(child, root, accum_rewards)

                # Oppdater min/max Q for normalisering
                for _, c in root.children.items():
                    if c.visit_count > 0:
                        q = c.q_value
                        min_q = min(min_q, q)
                        max_q = max(max_q, q)

        # Beregn policy fra visit counts
        policy = self._compute_policy(root, legal_actions)
        value = root.q_value if root.visit_count > 0 else float(init_value)
        self.last_root = root
        return policy, value

    def _tree_policy(self, node, min_q, max_q):
        """Traverser treet med PUCT til et leaf node.

        PUCT score = normalized_Q + c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child)

        Returns:
            (leaf_node, depth)
        """
        current = node
        depth = 0

        while current.is_expanded and current.children and depth < self.config.max_search_depth:
            best_action = None
            best_score = -float('inf')

            parent_visits = max(1, current.visit_count)

            for action, child in current.children.items():
                # Normaliser Q-verdi til [0, 1]
                q = child.q_value
                if max_q > min_q:
                    normalized_q = (q - min_q) / (max_q - min_q)
                else:
                    normalized_q = 0.0

                # PUCT score med normalisert Q
                exploration = (self.config.c_puct * child.prior *
                             math.sqrt(parent_visits) / (1 + child.visit_count))
                score = normalized_q + exploration

                if score > best_score:
                    best_score = score
                    best_action = action

            current = current.children[best_action]
            depth += 1

        return current, depth

    def _expand(self, node, psi_params, policy_override=None):
        """Utvid et leaf node ved å generere alle children.

        For hvert mulig action:
        - Bruk NNd til å beregne neste abstract state og reward
        - Bruk NNp for prior policy
        """
        if policy_override is not None:
            policy = policy_override
        else:
            policy, _ = self.trinet.prediction(psi_params, node.abstract_state)
            policy = np.array(policy)

        for action in range(self.num_actions):
            next_state, reward = self.trinet.dynamics(psi_params, node.abstract_state, action)

            child = MCTSNode(
                abstract_state=next_state,
                parent=node,
                action=action,
                reward=float(reward),
                prior=float(policy[action])
            )
            node.children[action] = child

        node.is_expanded = True

    def _rollout(self, node, remaining_depth, psi_params):
        """Utfør rollout fra et node med NNp og NNd.

        Følger pseudokoden DO_ROLLOUT fra oppgaven.
        Sampler actions fra policy og simulerer fremover.
        """
        sigma = node.abstract_state
        accum_rewards = []

        for d in range(remaining_depth):
            policy, value = self.trinet.prediction(psi_params, sigma)
            policy = np.array(policy)

            # Sample action fra policy
            policy = np.maximum(policy, 1e-8)
            policy = policy / np.sum(policy)
            action = np.random.choice(self.num_actions, p=policy)

            # Dynamics: neste state og reward
            sigma, r = self.trinet.dynamics(psi_params, sigma, action)
            accum_rewards.append(float(r))

        # Final value prediction
        _, final_value = self.trinet.prediction(psi_params, sigma)
        accum_rewards.append(float(final_value))

        return accum_rewards

    def _backpropagate(self, node, goal_node, rewards):
        """Backpropager rewards oppover i treet.

        Følger pseudokoden DO_BACKPROPAGATION fra oppgaven.
        Beregner diskontert sum og propagerer oppover med edge rewards.
        node.reward representerer belønningen fra å ta handlingen som førte til node.
        """
        gamma = self.config.discount_factor

        # Beregn diskontert sum av rollout rewards
        discounted_return = 0.0
        for i in reversed(range(len(rewards))):
            discounted_return = rewards[i] + gamma * discounted_return

        # Propager oppover fra node til goal_node
        current = node
        value = discounted_return

        while current is not None:
            current.visit_count += 1
            current.total_value += value

            if current == goal_node:
                break

            # Når vi går til parent, inkluder reward fra kanten (current.reward)
            # og diskonter verdien
            value = current.reward + gamma * value
            current = current.parent

    def _compute_policy(self, root, legal_actions):
        """Beregn policy fra visit counts i root.

        Bruker temperaturskalert softmax over visit counts:
        policy[a] = N(root, a)^(1/tau) / sum(N(root, a')^(1/tau))
        """
        visit_counts = np.zeros(self.num_actions, dtype=np.float32)

        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        tau = self.config.temperature
        if tau <= 0.01:
            policy = np.zeros(self.num_actions, dtype=np.float32)
            best_action = np.argmax(visit_counts)
            policy[best_action] = 1.0
        else:
            # Unngå overflow med store visit counts
            counts_temp = np.power(visit_counts + 1e-8, 1.0 / tau)
            total = np.sum(counts_temp)
            if total > 0:
                policy = counts_temp / total
            else:
                policy = np.ones(self.num_actions, dtype=np.float32) / self.num_actions

        return policy
