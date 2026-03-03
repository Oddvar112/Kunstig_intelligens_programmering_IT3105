class Config:
    """Sentral konfigurasjon for MuZero."""

    def __init__(self):
        # ==================== SPILLMILJØ ====================
        self.grid_size = 4

        # ==================== EPISODE LOOP ====================
        self.num_episodes = 16000        # Ne: totalt antall episoder
        self.steps_per_episode = 50     # Nes: maks steg per episode
        self.training_interval = 1      # It: tren Psi hver It episode

        # ==================== MCTS ====================
        self.num_mcts_searches = 100    # Ms: antall MCTS-søk
        self.max_search_depth = 5       # dmax: maks dybde i MCTS-tre
        self.discount_factor = 0.97     # gamma: diskonteringsfaktor
        self.c_puct = 1.0               # utforskingskonstant for PUCT/UCB
        self.dirichlet_alpha = 1.0      # Dirichlet noise (økt fra 0.5)
        self.dirichlet_epsilon = 0.5    # vekt for Dirichlet noise (økt fra 0.25)

        # Handlinger: UP=0, RIGHT=1, DOWN=2, LEFT=3
        self.num_actions = 4

        # ==================== NEVRALE NETTVERK ====================
        self.abstract_state_size = 32   # størrelse på abstract state
        self.lookback_q = 0             # q+1 states (0 for fullt observerbart spill)
        self.rollout_w = 5              # w steg fremover i BPTT

        # ==================== TRENING ====================
        self.learning_rate = 0.001
        self.minibatch_size = 32        # mbs: minibatch-størrelse

        # Temperatur for MCTS policy
        self.temperature = 1.5          # tau: høyere = mer utforsking
        self.temperature_decay = 0.999  # gradvis mer greedy

        # ==================== EVALUERING ====================
        self.eval_interval = 10         # evaluer policy hver N episode
        self.num_eval_episodes = 5      # antall eval-episoder

        # Beregn nettverksstørrelser
        self._compute_network_sizes()

    def _compute_network_sizes(self):
        """Beregn input/output-størrelser for nettverkene."""
        # State-representasjon:
        # [agent_row, agent_col, delta_row, delta_col, manhattan,
        #  at_top, at_bottom, at_left, at_right] = 9 elementer
        self.game_state_size = 9

        # NNr (Representation Network): states -> abstract state
        nnr_input = (self.lookback_q + 1) * self.game_state_size
        self.nnr_layers = [nnr_input, 128, 64, self.abstract_state_size]
        self.nnr_activations = ['relu', 'relu']

        # NNd (Dynamics Network): (abstract state, action) -> (next state, reward)
        nnd_input = self.abstract_state_size + self.num_actions
        nnd_output = self.abstract_state_size + 1
        self.nnd_layers = [nnd_input, 128, 64, nnd_output]
        self.nnd_activations = ['relu', 'relu']

        # NNp (Prediction Network): abstract state -> (policy, value)
        nnp_input = self.abstract_state_size
        nnp_output = self.num_actions + 1
        self.nnp_layers = [nnp_input, 128, 64, nnp_output]
        self.nnp_activations = ['relu', 'relu']

