"""
Demo-spiller for trent MuZero-agent.
Laster inn lagrede psi_params og spiller episoder med detaljert output.

Bruk: python play_demo.py [antall_episoder]
Eksempel: python play_demo.py 5
"""

import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from config import Config
from game_manager import GameStateManager
from trinet import TrinetManager


def play_demo(psi_params, config, num_episodes=3):
    game = GameStateManager(config)
    trinet = TrinetManager(config)

    action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
    action_arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    all_rewards = []
    all_collections = []

    for ep in range(num_episodes):
        state = game.reset()
        state_history = []
        total_reward = 0.0
        collections = 0
        steps_data = []

        print(f"\n{'='*50}")
        print(f"  EPISODE {ep+1}")
        print(f"{'='*50}")

        for k in range(config.steps_per_episode):
            state_array = game.state_to_array(state)
            state_history.append(state_array)

            # Lookback states
            q = config.lookback_q
            phi_k = []
            for i in range(q + 1):
                idx = k - q + i
                if idx < 0:
                    phi_k.append(np.zeros(config.game_state_size, dtype=np.float32))
                else:
                    phi_k.append(state_history[idx])

            phi_flat = np.concatenate(phi_k)
            sigma = trinet.representation(psi_params, jnp.array(phi_flat))
            policy, value = trinet.prediction(psi_params, sigma)
            policy = np.array(policy)

            # Velg beste handling (greedy)
            action = int(np.argmax(policy))

            # Vis tilstand
            agent_pos = state.agent_pos
            target_pos = state.target_pos
            print(f"\nSteg {k:2d} | Agent: {agent_pos} | Mål: {target_pos} | "
                  f"Action: {action_arrows[action]} {action_names[action]:5s} | "
                  f"Value: {float(value):.2f} | "
                  f"Policy: [{', '.join(f'{p:.3f}' for p in policy)}]")

            # Vis grid
            game.render(state)

            # Utfør handling
            next_state, reward = game.step(state, action)
            total_reward += reward

            if reward > 0.5:
                collections += 1
                print(f"  *** MÅL SAMLET! (+1.0) *** Total: {collections}")

            steps_data.append({
                'agent': agent_pos,
                'target': target_pos,
                'action': action_names[action],
                'reward': reward
            })

            state = next_state

        all_rewards.append(total_reward)
        all_collections.append(collections)

        print(f"\n{'─'*50}")
        print(f"  Episode {ep+1} ferdig!")
        print(f"  Total reward:  {total_reward:.2f}")
        print(f"  Mål samlet:    {collections}")
        print(f"  Snitt steg/mål: {config.steps_per_episode / max(1, collections):.1f}")
        print(f"{'─'*50}")

    # Oppsummering
    print(f"\n{'='*50}")
    print(f"  OPPSUMMERING ({num_episodes} episoder)")
    print(f"{'='*50}")
    print(f"  Gjennomsnittlig reward:     {np.mean(all_rewards):.2f}")
    print(f"  Standardavvik:              {np.std(all_rewards):.2f}")
    print(f"  Beste episode:              {max(all_rewards):.2f}")
    print(f"  Dårligste episode:          {min(all_rewards):.2f}")
    print(f"  Gjennomsnittlig mål samlet: {np.mean(all_collections):.1f}")
    print(f"{'='*50}")

    return steps_data  # Returnerer siste episode for evt. GIF


def main():
    num_episodes = 3
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            print(f"Ugyldig antall episoder: {sys.argv[1]}, bruker 3")

    # Last inn params
    try:
        with open('psi_params.pkl', 'rb') as f:
            psi_params = pickle.load(f)
        print("Lastet psi_params.pkl")
    except FileNotFoundError:
        print("FEIL: Finner ikke psi_params.pkl!")
        print("Kjør trening først med lagring av params.")
        sys.exit(1)

    config = Config()
    play_demo(psi_params, config, num_episodes)


if __name__ == '__main__':
    main()