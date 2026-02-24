import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # bruk non-interactive backend som standard
import matplotlib.pyplot as plt

from config import Config
from game_manager import GameStateManager
from trinet import TrinetManager
from mcts import UMCTS
from episode_buffer import EpisodeBuffer
from rl_manager import RLManager


def plot_results(reward_history, loss_history, config):
    """Visualiser treningsresultater."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Reward per episode
    ax1 = axes[0]
    ax1.plot(reward_history, alpha=0.3, color='blue', label='Reward per episode')

    # Glidende gjennomsnitt
    window = min(20, max(2, len(reward_history) // 5))
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(reward_history)), moving_avg,
                color='red', linewidth=2, label=f'Glidende gj.snitt ({window} ep)')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title(f'MuZero Trening - GridCollect ({config.grid_size}x{config.grid_size})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: BPTT Loss
    ax2 = axes[1]
    if loss_history:
        ax2.plot(loss_history, color='green', linewidth=1.5)
        ax2.set_xlabel('Treningssteg')
        ax2.set_ylabel('BPTT Loss')
        ax2.set_title('Trenings-loss over tid')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Ingen treningsdata enda', ha='center', va='center')

    plt.tight_layout()
    plt.savefig('muzero_results.png', dpi=150)
    print("Resultater lagret til muzero_results.png")
    plt.close()


def demo_gameplay(rl_manager, psi_params, num_episodes=3):
    """Demonstrer spilling med trent policy."""
    print("\n" + "="*50)
    print("DEMO: Spiller med trent policy (uten MCTS)")
    print("="*50)

    rewards = []
    for i in range(num_episodes):
        print(f"\n--- Demo episode {i+1} ---")
        reward = rl_manager.play_episode(psi_params, render=True)
        rewards.append(reward)

    print(f"\nGjennomsnittlig demo-reward: {np.mean(rewards):.2f}")


def compare_random_vs_trained(rl_manager, psi_params, num_episodes=20):
    """Sammenlign trent policy med tilfeldig spilling."""
    print("\n" + "="*50)
    print("Sammenligning: Tilfeldig vs Trent policy")
    print("="*50)

    # Tilfeldig spilling
    random_rewards = []
    game = rl_manager.game
    for _ in range(num_episodes):
        state = game.reset()
        total_reward = 0.0
        for _ in range(rl_manager.config.steps_per_episode):
            action = np.random.choice(rl_manager.config.num_actions)
            state, reward = game.step(state, action)
            total_reward += reward
        random_rewards.append(total_reward)

    # Trent policy
    trained_rewards = []
    for _ in range(num_episodes):
        reward = rl_manager.play_episode(psi_params, render=False)
        trained_rewards.append(reward)

    print(f"Tilfeldig spilling:  Gj.snitt={np.mean(random_rewards):7.2f}, "
          f"Std={np.std(random_rewards):5.2f}")
    print(f"Trent policy:        Gj.snitt={np.mean(trained_rewards):7.2f}, "
          f"Std={np.std(trained_rewards):5.2f}")

    improvement = np.mean(trained_rewards) - np.mean(random_rewards)
    print(f"Forbedring:          {improvement:+7.2f}")


def main():
    """Hovedprogram for MuZero GridCollect."""
    print("="*50)
    print("  MuZero Knockoff - GridCollect")
    print("  IT-3105 AI Programming")
    print("="*50)

    # Sjekk kommandolinje-argumenter
    config = Config()
    if len(sys.argv) > 1 and sys.argv[1] == '--small':
        config.set_small()
        print("\n[Kjorer med sma parametre for rask testing]")

    print(f"\nKonfigurasjon:")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Episoder: {config.num_episodes}")
    print(f"  Steg/episode: {config.steps_per_episode}")
    print(f"  MCTS sok: {config.num_mcts_searches}")
    print(f"  Max sokedybde: {config.max_search_depth}")
    print(f"  Abstract state storrelse: {config.abstract_state_size}")
    print(f"  Lookback q: {config.lookback_q}")
    print(f"  Roll-ahead w: {config.rollout_w}")
    print(f"  Laeringsrate: {config.learning_rate}")
    print(f"  Minibatch storrelse: {config.minibatch_size}")
    print(f"  Treningsintervall: {config.training_interval}")
    print()

    # Opprett alle komponenter
    game_manager = GameStateManager(config)
    trinet_manager = TrinetManager(config)
    mcts = UMCTS(config, trinet_manager)
    episode_buffer = EpisodeBuffer(config)
    rl_manager = RLManager(config, game_manager, trinet_manager, mcts, episode_buffer)

    # Kjor trening
    psi_params, reward_history, loss_history = rl_manager.run()

    # Visualiser resultater
    plot_results(reward_history, loss_history, config)

    # Sammenlign med tilfeldig spilling
    compare_random_vs_trained(rl_manager, psi_params)

    # Demo med trent policy
    demo_gameplay(rl_manager, psi_params, num_episodes=1)


if __name__ == "__main__":
    main()
