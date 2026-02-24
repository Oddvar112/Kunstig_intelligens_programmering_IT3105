import numpy as np


class EpisodeBuffer:
    """Episode Buffer for lagring av episodedata til BPTT-trening.

    Lagrer for hver episode:
    - states: sekvens av game states (som flat arrays)
    - actions: sekvens av handlinger
    - rewards: sekvens av belønninger
    - policies: sekvens av MCTS-policies
    - values: sekvens av estimerte state-verdier
    """

    def __init__(self, config):
        self.config = config
        self.episodes = []
        self.max_size = 200  # behold flere episoder for mer stabil trening

    def add_episode(self, episode_data):
        """Legg til en episode i bufferet.

        Args:
            episode_data: liste av [state_array, value, policy, action, reward]
        """
        self.episodes.append(episode_data)
        # Fjern eldste episode hvis bufferet er fullt
        if len(self.episodes) > self.max_size:
            self.episodes.pop(0)

    def sample_minibatch(self):
        """Trekk en tilfeldig minibatch for BPTT-trening.

        For hver sample:
        1. Velg tilfeldig episode og tilfeldig state index k
        2. Hent states S = {s_{k-q}, ..., s_k} (pad med blanke)
        3. Hent actions A = {a_{k+1}, ..., a_{k+w}}
        4. Hent target policies = {pi_k, ..., pi_{k+w}}
        5. Hent target values = {v_k, ..., v_{k+w}}
        6. Hent target rewards = {r_{k+1}, ..., r_{k+w}}

        Returns:
            liste av (states_flat, actions, target_policies, target_values, target_rewards)
            eller None hvis ikke nok data
        """
        if len(self.episodes) == 0:
            return None

        q = self.config.lookback_q
        w = self.config.rollout_w
        state_size = self.config.game_state_size
        mbs = self.config.minibatch_size

        minibatch = []

        for _ in range(mbs):
            # Velg tilfeldig episode
            ep_idx = np.random.randint(len(self.episodes))
            episode = self.episodes[ep_idx]

            ep_len = len(episode)
            if ep_len < w + 1:
                continue

            # Velg tilfeldig state index k slik at vi har nok data fremover
            max_k = ep_len - w - 1
            if max_k < 0:
                continue
            k = np.random.randint(0, max_k + 1)

            # episode_data[i] = [state_array, value, policy, action, reward]
            # Hent lookback states: s_{k-q}, ..., s_k
            lookback_states = []
            for i in range(q + 1):
                idx = k - q + i
                if idx < 0:
                    # Pad med blank state
                    lookback_states.append(np.zeros(state_size, dtype=np.float32))
                else:
                    lookback_states.append(episode[idx][0])  # state_array

            states_flat = np.concatenate(lookback_states)

            # Hent roll-ahead actions: a_k, a_{k+1}, ..., a_{k+w-1}
            actions = []
            for j in range(w):
                idx = k + j
                if idx < ep_len:
                    actions.append(episode[idx][3])  # action
                else:
                    actions.append(0)  # default action

            # Hent target policies: pi_k, ..., pi_{k+w}
            target_policies = []
            for j in range(w + 1):
                idx = k + j
                if idx < ep_len:
                    target_policies.append(episode[idx][2])  # policy
                else:
                    # Uniform policy som fallback
                    target_policies.append(
                        np.ones(self.config.num_actions, dtype=np.float32) / self.config.num_actions
                    )

            # Hent target values: v_k, ..., v_{k+w}
            target_values = []
            for j in range(w + 1):
                idx = k + j
                if idx < ep_len:
                    target_values.append(episode[idx][1])  # value
                else:
                    target_values.append(0.0)

            # Hent target rewards: r_k, r_{k+1}, ..., r_{k+w-1} (reward from each transition)
            target_rewards = []
            for j in range(w):
                idx = k + j
                if idx < ep_len:
                    target_rewards.append(episode[idx][4])  # reward
                else:
                    target_rewards.append(0.0)

            minibatch.append((
                states_flat,
                actions,
                target_policies,
                target_values,
                target_rewards
            ))

        if len(minibatch) == 0:
            return None

        return minibatch

    def __len__(self):
        return len(self.episodes)
