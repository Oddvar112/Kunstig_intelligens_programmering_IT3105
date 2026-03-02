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


def plot_results(reward_history, loss_history, config, training_stats=None):
    """Visualiser treningsresultater med flere plots."""
    
    # === PLOT 1: Kombinert oversikt (original) ===
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
    
    # === PLOT 2: Kun reward per episode ===
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(reward_history, color='blue', alpha=0.7, linewidth=1)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_per_episode.png', dpi=150)
    print("Lagret: reward_per_episode.png")
    plt.close()
    
    # === PLOT 3: Kun glidende gjennomsnitt ===
    fig, ax = plt.subplots(figsize=(12, 5))
    windows = [10, 25, 50, 100]
    colors = ['blue', 'green', 'orange', 'red']
    for w, c in zip(windows, colors):
        if len(reward_history) >= w:
            ma = np.convolve(reward_history, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(reward_history)), ma, 
                   color=c, linewidth=2, label=f'MA({w})', alpha=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Gjennomsnittlig Reward')
    ax.set_title('Glidende Gjennomsnitt av Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_moving_average.png', dpi=150)
    print("Lagret: reward_moving_average.png")
    plt.close()
    
    # === PLOT 4: Kumulativ reward ===
    fig, ax = plt.subplots(figsize=(12, 5))
    cumulative = np.cumsum(reward_history)
    ax.plot(cumulative, color='purple', linewidth=2)
    ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Kumulativ Reward')
    ax.set_title('Kumulativ Reward over Episoder')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_reward.png', dpi=150)
    print("Lagret: cumulative_reward.png")
    plt.close()
    
    # === PLOT 5: Reward distribusjon (histogram) ===
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(reward_history, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(reward_history), color='red', linestyle='--', 
               linewidth=2, label=f'Gjennomsnitt: {np.mean(reward_history):.2f}')
    ax.axvline(np.median(reward_history), color='orange', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(reward_history):.2f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Antall Episoder')
    ax.set_title('Fordeling av Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_distribution.png', dpi=150)
    print("Lagret: reward_distribution.png")
    plt.close()
    
    # === PLOT 6: Episodevis forbedring ===
    if len(reward_history) > 1:
        fig, ax = plt.subplots(figsize=(12, 5))
        improvement = np.diff(reward_history)
        colors = ['green' if x >= 0 else 'red' for x in improvement]
        ax.bar(range(len(improvement)), improvement, color=colors, alpha=0.7, width=1.0)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Endring fra forrige episode')
        ax.set_title('Episodevis Reward-endring')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('episode_improvement.png', dpi=150)
        print("Lagret: episode_improvement.png")
        plt.close()
    
    # === PLOT 7: Rolling standardavvik (stabilitetsanalyse) ===
    fig, ax = plt.subplots(figsize=(12, 5))
    window_std = 20
    if len(reward_history) >= window_std:
        rolling_std = []
        for i in range(window_std, len(reward_history)+1):
            rolling_std.append(np.std(reward_history[i-window_std:i]))
        ax.plot(range(window_std-1, len(reward_history)), rolling_std, 
               color='darkorange', linewidth=2)
        ax.fill_between(range(window_std-1, len(reward_history)), rolling_std, 
                        alpha=0.3, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Standardavvik (siste 20 ep)')
    ax.set_title('Rolling Standardavvik - Policy Stabilitet')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reward_stability.png', dpi=150)
    print("Lagret: reward_stability.png")
    plt.close()
    
    # === PLOT 8: Trenings-loss detaljert ===
    if loss_history and len(loss_history) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss over tid
        ax1 = axes[0]
        ax1.plot(loss_history, color='green', linewidth=1.5, alpha=0.7)
        if len(loss_history) >= 10:
            ma = np.convolve(loss_history, np.ones(10)/10, mode='valid')
            ax1.plot(range(9, len(loss_history)), ma, color='darkgreen', 
                    linewidth=2, label='MA(10)')
        ax1.set_xlabel('Treningssteg')
        ax1.set_ylabel('Loss')
        ax1.set_title('BPTT Loss over tid')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss histogram
        ax2 = axes[1]
        ax2.hist(loss_history, bins=30, color='green', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(loss_history), color='red', linestyle='--', 
                   linewidth=2, label=f'Gj.snitt: {np.mean(loss_history):.4f}')
        ax2.set_xlabel('Loss')
        ax2.set_ylabel('Frekvens')
        ax2.set_title('Fordeling av Loss-verdier')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('loss_detailed.png', dpi=150)
        print("Lagret: loss_detailed.png")
        plt.close()
    
    # === PLOT 9-11: Entropy, Max Prob, Collections (hvis tilgjengelig) ===
    if training_stats:
        entropy_history = training_stats.get('entropy_history', [])
        max_prob_history = training_stats.get('max_prob_history', [])
        collections_history = training_stats.get('collections_history', [])
        
        # Entropy plot
        if entropy_history:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(entropy_history, color='purple', linewidth=1.5, alpha=0.7)
            if len(entropy_history) >= 20:
                ma = np.convolve(entropy_history, np.ones(20)/20, mode='valid')
                ax.plot(range(19, len(entropy_history)), ma, 
                       color='darkviolet', linewidth=2, label='MA(20)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Policy Entropy')
            ax.set_title('Policy Entropy over tid (høyere = mer utforsking)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('policy_entropy.png', dpi=150)
            print("Lagret: policy_entropy.png")
            plt.close()
        
        # Max probability plot
        if max_prob_history:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(max_prob_history, color='teal', linewidth=1.5, alpha=0.7)
            if len(max_prob_history) >= 20:
                ma = np.convolve(max_prob_history, np.ones(20)/20, mode='valid')
                ax.plot(range(19, len(max_prob_history)), ma, 
                       color='darkcyan', linewidth=2, label='MA(20)')
            ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Uniform (0.25)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Gjennomsnittlig Max Policy-sannsynlighet')
            ax.set_title('Policy Konfidens over tid')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig('policy_confidence.png', dpi=150)
            print("Lagret: policy_confidence.png")
            plt.close()
        
        # Collections per episode
        if collections_history:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(collections_history, color='gold', linewidth=1.5, alpha=0.7)
            if len(collections_history) >= 20:
                ma = np.convolve(collections_history, np.ones(20)/20, mode='valid')
                ax.plot(range(19, len(collections_history)), ma, 
                       color='darkorange', linewidth=2, label='MA(20)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Antall mål samlet')
            ax.set_title('Mål Samlet per Episode')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('collections_per_episode.png', dpi=150)
            print("Lagret: collections_per_episode.png")
            plt.close()
    
    # === PLOT 12: Stor dashboard-figur ===
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Reward med MA
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(reward_history, alpha=0.3, color='blue', label='Reward')
    if len(reward_history) >= 20:
        ma = np.convolve(reward_history, np.ones(20)/20, mode='valid')
        ax1.plot(range(19, len(reward_history)), ma, color='red', linewidth=2, label='MA(20)')
    ax1.set_title('Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Kumulativ reward
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(np.cumsum(reward_history), color='purple', linewidth=2)
    ax2.set_title('Kumulativ Reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Kumulativ Reward')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Loss
    ax3 = fig.add_subplot(3, 2, 3)
    if loss_history:
        ax3.plot(loss_history, color='green', linewidth=1.5)
    ax3.set_title('Trenings-Loss')
    ax3.set_xlabel('Treningssteg')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Reward histogram
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.hist(reward_history, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    ax4.axvline(np.mean(reward_history), color='red', linestyle='--', linewidth=2)
    ax4.set_title(f'Reward Fordeling (gj.snitt: {np.mean(reward_history):.2f})')
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Frekvens')
    ax4.grid(True, alpha=0.3)
    
    # Subplot 5: Entropy (hvis tilgjengelig)
    ax5 = fig.add_subplot(3, 2, 5)
    if training_stats and training_stats.get('entropy_history'):
        ax5.plot(training_stats['entropy_history'], color='purple', linewidth=1.5)
        ax5.set_title('Policy Entropy')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Entropy')
    else:
        ax5.text(0.5, 0.5, 'Entropy data ikke tilgjengelig', ha='center', va='center')
    ax5.grid(True, alpha=0.3)
    
    # Subplot 6: Collections (hvis tilgjengelig)
    ax6 = fig.add_subplot(3, 2, 6)
    if training_stats and training_stats.get('collections_history'):
        ax6.plot(training_stats['collections_history'], color='gold', linewidth=1.5)
        ax6.set_title('Mål Samlet per Episode')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Antall')
    else:
        ax6.text(0.5, 0.5, 'Collections data ikke tilgjengelig', ha='center', va='center')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'MuZero GridCollect Dashboard ({config.grid_size}x{config.grid_size})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=150)
    print("Lagret: dashboard.png")
    plt.close()
    
    # === PLOT 13: Kun glidende gjennomsnitt (enkelt plot) ===
    fig, ax = plt.subplots(figsize=(12, 5))
    window = min(20, max(2, len(reward_history) // 5))
    if len(reward_history) >= window:
        moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(reward_history)), moving_avg,
               color='red', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Glidende Gjennomsnitt Reward')
    ax.set_title(f'Glidende Gjennomsnitt ({window} episoder)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('moving_average_only.png', dpi=150)
    print("Lagret: moving_average_only.png")
    plt.close()
    
    print("\n=== Alle plots generert ===")
    print("Plots lagret:")
    print("  1. muzero_results.png - Kombinert oversikt (original)")
    print("  2. reward_per_episode.png - Kun reward per episode")
    print("  3. reward_moving_average.png - Glidende gjennomsnitt (10, 25, 50, 100)")
    print("  4. cumulative_reward.png - Kumulativ reward")
    print("  5. reward_distribution.png - Histogram av rewards")
    print("  6. episode_improvement.png - Endring mellom episoder")
    print("  7. reward_stability.png - Rolling standardavvik")
    print("  8. loss_detailed.png - Detaljert loss-analyse")
    print("  9. policy_entropy.png - Policy entropy over tid")
    print(" 10. policy_confidence.png - Policy konfidens over tid")
    print(" 11. collections_per_episode.png - Mål samlet per episode")
    print(" 12. dashboard.png - Dashboard med alle viktige metrics")
    print(" 13. moving_average_only.png - Kun glidende gjennomsnitt")


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
    psi_params, reward_history, loss_history, training_stats = rl_manager.run()

    # Visualiser resultater
    plot_results(reward_history, loss_history, config, training_stats)

    # Sammenlign med tilfeldig spilling
    compare_random_vs_trained(rl_manager, psi_params)

    # Demo med trent policy
    demo_gameplay(rl_manager, psi_params, num_episodes=1)


if __name__ == "__main__":
    main()
