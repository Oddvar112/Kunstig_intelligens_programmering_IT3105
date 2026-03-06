"""Reinforcement Learning Manager for MuZero.

Denne modulen implementerer hovedtreningsløkken for MuZero-algoritmen.
RLManager koordinerer spilling av episoder, MCTS-søk, datainnsamling,
og trening av de tre nettverkene.

Følger EPISODE_LOOP pseudokoden fra oppgaven.
"""

import numpy as np
import jax
import jax.numpy as jnp
import os
import pickle
from datetime import datetime


class RLManager:
    """
    Reinforcement Learning Manager - koordinerer MuZero-trening.
    Implementerer hovedløkken som:
    1. Initialiserer nettverksparametere
    2. Kjører episoder med MCTS-guidet spilling
    3. Samler treningsdata i buffer
    4. Trener nettverkene periodisk
    """


    def __init__(self, config, game_manager, trinet_manager, mcts, episode_buffer):
        self.config = config
        self.game = game_manager
        self.trinet = trinet_manager
        self.mcts = mcts
        self.buffer = episode_buffer
        
        # Setup logging
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_{timestamp}.log")
        self.detailed_log_file = os.path.join(self.log_dir, f"detailed_{timestamp}.log")

    def run(self):
        """Kjør MuZero episode loop.

        Følger pseudokoden:
        1. Init Psi tilfeldig
        2. For episode in range(Ne):
           a. Reset spill
           b. For k in range(Nes): kjør MCTS, sample action, step spill
           c. Lagre episode data
           d. Tren Psi periodisk
        3. Returner trenet Psi

        Returns:
            (psi_params, reward_history, loss_history)
        """
        # 1. Initialiser parametre tilfeldig
        key = jax.random.PRNGKey(42)
        psi_params = self.trinet.init_params(key)

        reward_history = []
        loss_history = []
        entropy_history = []
        max_prob_history = []
        collections_history = []

        self._log_config()

        for episode in range(self.config.num_episodes):
            # a. Reset spill
            state = self.game.reset()
            episode_data = []
            state_history = []  # for lookback
            total_reward = 0.0

            # Oppdater temperatur for MCTS-policy (decay over tid)
            self.mcts.config.temperature = max(
                0.1,
                self.config.temperature * (self.config.temperature_decay ** episode)
            )

            # b. Kjør episode
            for k in range(self.config.steps_per_episode):
                # Lagre nåværende state
                state_array = self.game.state_to_array(state)
                state_history.append(state_array)

                # Samle q+1 states (phi_k) med padding for tidlige steg
                phi_k = self._gather_lookback_states(state_history, k)

                # Opprett abstract state via NNr
                phi_flat = np.concatenate(phi_k)
                sigma_k = self.trinet.representation(psi_params, jnp.array(phi_flat))

                # Kjør MCTS-søk
                legal_actions = self.game.get_legal_actions(state)
                policy_k, value_k = self.mcts.search(psi_params, sigma_k, legal_actions)
                
                # Detaljert MCTS logging (hver 50. episode, første 3 steg)
                if episode % 50 == 0 and k < 3:
                    self._log_mcts_details(episode, k, state, policy_k, value_k)

                # Sample action fra policy
                policy_k = np.array(policy_k, dtype=np.float64)
                policy_k = np.maximum(policy_k, 0)
                policy_sum = np.sum(policy_k)
                if policy_sum > 0:
                    policy_k = policy_k / policy_sum
                else:
                    policy_k = np.ones(self.config.num_actions) / self.config.num_actions

                action = np.random.choice(self.config.num_actions, p=policy_k)

                # Step spill
                next_state, reward = self.game.step(state, action)
                total_reward += reward

                # Lagre episode data: [state_array, value, policy, action, reward]
                episode_data.append([
                    state_array,
                    float(value_k),
                    policy_k.astype(np.float32),
                    int(action),
                    float(reward)
                ])

                state = next_state

         
            # Log episode summary og samle metrics
            avg_entropy, avg_max_prob, collections = self._log_episode_summary(episode, episode_data, total_reward)
            entropy_history.append(avg_entropy)
            max_prob_history.append(avg_max_prob)
            collections_history.append(collections)
            
            # Buffer statistikk
            self._log_buffer_stats(episode)

            # c. Legg til episode i buffer
            self.buffer.add_episode(episode_data)
            reward_history.append(total_reward)

            # d. Tren Psi periodisk - multiple treningssteg per intervall
            # Warm-up: ikke tren de første 20 episodene for å samle meningsfulle data
            avg_loss = 0.0
            if episode > 20 and episode % self.config.training_interval == 0:
                if len(self.buffer) >= 1:
                    # Kjør flere treningssteg per episode
                    num_train_steps = 12
                    total_loss = 0.0
                    for _ in range(num_train_steps):
                        psi_params, step_loss = self.trinet.train_step(psi_params, self.buffer)
                        total_loss += step_loss
                    avg_loss = total_loss / num_train_steps
                    loss_history.append(avg_loss)
                    self._log_training(num_train_steps, avg_loss)
            
            # Detaljert diagnostikk hver 25. episode 
            if episode > 0 and episode % 25 == 0:
                self._log_diagnostics(episode, psi_params)

            # Logg fremgang
            self._log_progress(episode, total_reward, avg_loss, reward_history)

            # Periodisk eval med ren policy (uten MCTS) for å fange kollaps
            if episode > 0 and episode % self.config.eval_interval == 0:
                self._log_eval(psi_params)

        self._log_training_complete(psi_params, reward_history, loss_history)
        
        # Returner alle metrics for plotting
        training_stats = {
            'entropy_history': entropy_history,
            'max_prob_history': max_prob_history,
            'collections_history': collections_history
        }
        
        # Lagre psi_params til fil
        with open('psi_params.pkl', 'wb') as f:
            pickle.dump(psi_params, f)
        
        return psi_params, reward_history, loss_history, training_stats

    def _gather_lookback_states(self, state_history, k):
        """
        Samle lookback states for representasjonsinput.
        Henter de siste q+1 states fra historikk, med padding
        for tidlige steg hvor det ikke finnes nok historikk.
        """
        q = self.config.lookback_q
        states = []

        for i in range(q + 1):
            idx = k - q + i
            if idx < 0:
                # Pad med blank state
                states.append(self.game.blank_state_array())
            else:
                states.append(state_history[idx])

        return states

    def play_episode(self, psi_params, render=False):
        """
        Spill en episode med ren policy (uten MCTS).
        Brukes for evaluering av lært policy.
        """
        state = self.game.reset()
        state_history = []
        total_reward = 0.0

        for k in range(self.config.steps_per_episode):
            state_array = self.game.state_to_array(state)
            state_history.append(state_array)
            phi_k = self._gather_lookback_states(state_history, k)
            phi_flat = np.concatenate(phi_k)
            sigma = self.trinet.representation(psi_params, jnp.array(phi_flat))
            policy, value = self.trinet.prediction(psi_params, sigma)
            policy = np.array(policy)

            # Bruk temperatur 0.1 for å unngå ren argmax som kan være ustabil
            policy = np.maximum(policy, 1e-8)
            policy_temp = np.power(policy, 1.0 / 0.5)
            policy_temp = policy_temp / np.sum(policy_temp)
            action = np.random.choice(self.config.num_actions, p=policy_temp)

            if render:
                print(f"\nSteg {k}, Policy: {policy}, Value: {float(value):.3f}, Action: {action}")
                self.game.render(state)

            # Step
            next_state, reward = self.game.step(state, action)
            total_reward += reward
            state = next_state

        if render:
            print(f"\nTotal reward: {total_reward:.2f}")

        return total_reward

    # ==================== LOGGING ====================
    # Disse metodene brukes for feilsøkning og monitorering under trening

    def _log(self, message, detailed=False):
        """Skriv til loggfil."""
        log_path = self.detailed_log_file if detailed else self.log_file
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def _log_config(self):
        """Logger konfigurasjon ved oppstart."""
        print("Starter MuZero trening...")
        print(f"Episoder: {self.config.num_episodes}, Steg/episode: {self.config.steps_per_episode}")
        print(f"MCTS søk: {self.config.num_mcts_searches}, Max dybde: {self.config.max_search_depth}")
        print(f"Logger til: {self.log_file}")
        print()
        
        self._log("=" * 60)
        self._log(f"MuZero Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log("=" * 60)
        self._log(f"Episodes: {self.config.num_episodes}")
        self._log(f"Steps/episode: {self.config.steps_per_episode}")
        self._log(f"MCTS searches: {self.config.num_mcts_searches}")
        self._log(f"Max search depth: {self.config.max_search_depth}")
        self._log(f"Discount factor: {self.config.discount_factor}")
        self._log(f"Learning rate: {self.config.learning_rate}")
        self._log(f"Temperature: {self.config.temperature} (decay: {self.config.temperature_decay})")
        self._log(f"Buffer size: {self.buffer.max_size}")
        self._log(f"Minibatch size: {self.config.minibatch_size}")
        self._log(f"Abstract state size: {self.config.abstract_state_size}")
        self._log(f"c_puct: {self.config.c_puct}")
        self._log(f"Dirichlet alpha: {self.config.dirichlet_alpha}, epsilon: {self.config.dirichlet_epsilon}")
        self._log("=" * 60 + "\n")
    
    def _log_mcts_details(self, episode, k, state, policy_k, value_k):
        """Logger detaljert MCTS-info for debugging."""
        agent_pos = state.agent_pos if hasattr(state, 'agent_pos') else '?'
        target_pos = state.target_pos if hasattr(state, 'target_pos') else '?'
        self._log(f"\n  [Ep {episode}, Step {k}] Agent: {agent_pos}, Target: {target_pos}", detailed=True)
        self._log(f"    MCTS policy: {np.array(policy_k).round(3)}", detailed=True)
        self._log(f"    MCTS value: {value_k:.4f}", detailed=True)
        self._log(f"    Temperature: {self.mcts.config.temperature:.4f}", detailed=True)
        
        if hasattr(self.mcts, 'last_root') and self.mcts.last_root:
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            self._log("    MCTS Tree:", detailed=True)
            for a, child in self.mcts.last_root.children.items():
                self._log(f"      {action_names[a]}: visits={child.visit_count}, Q={child.q_value:.4f}, prior={child.prior:.3f}", detailed=True)
    
    def _log_episode_summary(self, episode, episode_data, total_reward):
        """Logger oppsummering etter hver episode."""
        values = [e[1] for e in episode_data]
        rewards = [e[4] for e in episode_data]
        policies = [e[2] for e in episode_data]
        actions_taken = [e[3] for e in episode_data]
        
        avg_entropy = np.mean([-np.sum(p * np.log(p + 1e-8)) for p in policies])
        avg_max_prob = np.mean([np.max(p) for p in policies])
        action_counts = np.bincount(actions_taken, minlength=4)
        collections = sum(1 for r in rewards if r > 0.5)
        
        if episode % 10 == 0:
            print(f"  Value range: {min(values):.2f} to {max(values):.2f}")
            print(f"Values: min={min(values):.2f}, max={max(values):.2f}, avg={sum(values)/len(values):.2f}")
            print(f"Rewards: sum={sum(rewards):.2f}, Policy entropy={avg_entropy:.3f}, Avg max prob={avg_max_prob:.3f}")
        
        self._log(f"\nEpisode {episode}:")
        self._log(f"  Total reward: {total_reward:.2f}")
        self._log(f"  Collections: {collections}")
        self._log(f"  Value range: [{min(values):.3f}, {max(values):.3f}], avg={np.mean(values):.3f}")
        self._log(f"  Reward sum: {sum(rewards):.2f}")
        self._log(f"  Policy entropy: {avg_entropy:.4f} (higher=more exploration)")
        self._log(f"  Avg max policy prob: {avg_max_prob:.3f}")
        self._log(f"  Actions: UP={action_counts[0]}, RIGHT={action_counts[1]}, DOWN={action_counts[2]}, LEFT={action_counts[3]}")
        self._log(f"  Temperature: {self.mcts.config.temperature:.4f}")
        
        return avg_entropy, avg_max_prob, collections
    
    def _log_buffer_stats(self, episode):
        """Logger buffer-statistikk."""
        if episode % 50 != 0 or len(self.buffer) == 0:
            return
            
        all_rewards = []
        all_values = []
        for ep_data in self.buffer.episodes[-min(50, len(self.buffer.episodes)):]:
            for step in ep_data:
                all_rewards.append(step[4])
                all_values.append(step[1])
        
        self._log(f"  Buffer stats (last 50 eps): reward_mean={np.mean(all_rewards):.3f}, value_mean={np.mean(all_values):.3f}")
        self._log(f"    reward_range=[{min(all_rewards):.3f}, {max(all_rewards):.3f}], value_range=[{min(all_values):.3f}, {max(all_values):.3f}]")
        
        positive_rewards = sum(1 for r in all_rewards if r > 0)
        negative_rewards = sum(1 for r in all_rewards if r < 0)
        self._log(f"    positive_rewards={positive_rewards}, negative_rewards={negative_rewards}, ratio={positive_rewards/(negative_rewards+1):.3f}")
    
    def _log_training(self, num_train_steps, avg_loss):
        """Logger treningsstatistikk."""
        self._log(f"  Training: {num_train_steps} steps, avg_loss={avg_loss:.4f}")
        
        if hasattr(self.trinet, 'last_diagnostics'):
            diag = self.trinet.last_diagnostics
            self._log(f"  Gradients: norm={diag['grad_norm']:.4f}, clipped={diag['grad_norm_clipped']:.4f}, clip_factor={diag['clip_factor']:.3f}")
    
    def _log_diagnostics(self, episode, psi_params):
        """Logger detaljert diagnostikk."""
        diag = self.trinet.compute_diagnostics(psi_params, self.buffer)
        if not diag:
            return
            
        self._log(f"\n  === DIAGNOSTICS (Episode {episode}) ===")
        self._log(f"  Loss breakdown:")
        self._log(f"    Policy loss: {diag['policy_loss']:.4f}")
        self._log(f"    Value loss: {diag['value_loss']:.4f}")
        self._log(f"    Reward loss: {diag['reward_loss']:.4f}")
        self._log(f"    Entropy: {diag['entropy']:.4f}")
        
        self._log(f"  Value predictions vs targets:")
        pred_v = diag['pred_values'][:5]
        targ_v = diag['target_values'][:5]
        self._log(f"    Pred:   {[f'{v:.3f}' for v in pred_v]}")
        self._log(f"    Target: {[f'{v:.3f}' for v in targ_v]}")
        
        self._log(f"  Reward predictions vs targets:")
        pred_r = diag['pred_rewards'][:5]
        targ_r = diag['target_rewards'][:5]
        self._log(f"    Pred:   {[f'{v:.3f}' for v in pred_r]}")
        self._log(f"    Target: {[f'{v:.3f}' for v in targ_r]}")
        
        self._log(f"  Policy predictions (sample):")
        if diag['pred_policies']:
            self._log(f"    Pred:   {diag['pred_policies'][0].round(3)}")
            self._log(f"    Target: {diag['target_policies'][0].round(3)}")
        
        self._log(f"  Abstract state statistics:")
        self._log(f"    Mean: {diag['abstract_state_mean']:.4f}")
        self._log(f"    Std: {diag['abstract_state_std']:.4f}")
        self._log(f"    Range: [{diag['abstract_state_min']:.4f}, {diag['abstract_state_max']:.4f}]")
        self._log(f"    Diversity (pairwise dist): {diag['abstract_state_diversity']:.4f}")
        
        if diag['abstract_state_diversity'] < 0.1:
            self._log(f"  ⚠️ WARNING: Abstract state diversity very low - possible representation collapse!")
        
        avg_max_policy = np.mean([np.max(p) for p in diag['pred_policies']])
        if avg_max_policy > 0.95:
            self._log(f"  ⚠️ WARNING: Policy too deterministic (avg max prob={avg_max_policy:.3f})")
        
        self._log(f"  =======================\n")
    
    def _log_progress(self, episode, total_reward, avg_loss, reward_history):
        """Logger fremgang underveis."""
        if episode % 10 != 0 and episode != self.config.num_episodes - 1:
            return
            
        avg_recent = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
        print(f"Episode {episode:4d}/{self.config.num_episodes}: "
              f"Reward={total_reward:7.2f}, "
              f"Avg(10)={avg_recent:7.2f}, "
              f"Loss={avg_loss:.4f}, "
              f"Buffer={len(self.buffer)}")
    
    def _log_eval(self, psi_params):
        """Evaluerer og logger med ren policy."""
        eval_rewards = []
        for _ in range(self.config.num_eval_episodes):
            r = self.play_episode(psi_params, render=False)
            eval_rewards.append(r)
        eval_mean = np.mean(eval_rewards)
        eval_std = np.std(eval_rewards)
        print(f"  -> Eval (ren policy): Gj.snitt={eval_mean:.2f}")
        self._log(f"  EVAL: mean={eval_mean:.2f}, std={eval_std:.2f}, rewards={eval_rewards}")
    
    def _log_training_complete(self, psi_params, reward_history, loss_history):
        """Logger avsluttende oppsummering."""
        print("\nTrening fullført!")
        
        self._log("\n" + "=" * 60)
        self._log("TRAINING COMPLETE")
        self._log("=" * 60)
        self._log(f"Final avg reward (last 50): {np.mean(reward_history[-50:]):.2f}")
        self._log(f"Best reward: {max(reward_history):.2f}")
        self._log(f"Worst reward: {min(reward_history):.2f}")
        self._log(f"Final loss: {loss_history[-1] if loss_history else 'N/A'}")
        self._log(f"Total episodes: {len(reward_history)}")
        
        if len(loss_history) >= 10:
            self._log(f"\nLoss progression:")
            chunk_size = max(1, len(loss_history) // 10)
            for i in range(0, len(loss_history), chunk_size):
                chunk = loss_history[i:i+chunk_size]
                self._log(f"  Steps {i+1}-{i+len(chunk)}: avg_loss={np.mean(chunk):.4f}")
        
        final_diag = self.trinet.compute_diagnostics(psi_params, self.buffer)
        if final_diag:
            self._log(f"\nFinal network diagnostics:")
            self._log(f"  Policy loss: {final_diag['policy_loss']:.4f}")
            self._log(f"  Value loss: {final_diag['value_loss']:.4f}")
            self._log(f"  Reward loss: {final_diag['reward_loss']:.4f}")
            self._log(f"  Abstract state diversity: {final_diag['abstract_state_diversity']:.4f}")
            
            value_errors = [abs(p - t) for p, t in zip(final_diag['pred_values'], final_diag['target_values'])]
            self._log(f"  Value prediction error: mean={np.mean(value_errors):.4f}, max={np.max(value_errors):.4f}")
            
            reward_errors = [abs(p - t) for p, t in zip(final_diag['pred_rewards'], final_diag['target_rewards'])]
            self._log(f"  Reward prediction error: mean={np.mean(reward_errors):.4f}, max={np.max(reward_errors):.4f}")
        
        if len(reward_history) >= 100:
            self._log(f"\nReward progression:")
            self._log(f"  Episodes 1-50: avg={np.mean(reward_history[:50]):.2f}")
            self._log(f"  Episodes 51-100: avg={np.mean(reward_history[50:100]):.2f}")
            for i in range(100, len(reward_history), 100):
                end = min(i + 100, len(reward_history))
                self._log(f"  Episodes {i+1}-{end}: avg={np.mean(reward_history[i:end]):.2f}")
        
        print(f"Logg skrevet til: {self.log_file}")
        print(f"Detaljert logg: {self.detailed_log_file}")
