class Config:
    def __init__(self):
        # GridCollect spillparametre
        self.grid_size = 4

        # MuZero episode loop
        self.num_episodes = 1000       # Ne
        self.steps_per_episode = 50   # Nes
        self.training_interval = 1    # It - tren Psi hver It episode

        # u-MCTS
        self.num_mcts_searches = 100   # Ms
        self.max_search_depth = 5     # dmax
        self.discount_factor = 0.97   # gamma
        self.c_puct = 1.0             # utforskingskonstant for PUCT/UCB
        self.dirichlet_alpha = 1.0    # opp fra 0.5
        self.dirichlet_epsilon = 0.5  # opp fra 0.25

        # Antall handlinger: UP=0, RIGHT=1, DOWN=2, LEFT=3
        self.num_actions = 4

        # Nevrale nettverk
        self.abstract_state_size = 32  # større abstract state for bedre representasjon
        self.lookback_q = 0           # q+1 states - 0 er riktig for fullt observerbart spill
        self.rollout_w = 5            # w steg fremover i BPTT (match max_search_depth)

        # Trening
        self.learning_rate = 0.001     
        self.minibatch_size = 32      # mbs

        # Temperatur for MCTS policy
        self.temperature = 1.5        # tau - høyere = mer utforsking
        self.temperature_decay = 0.998 # gradvis mer greedy (saktere decay for bedre exploration)

        # Visualisering
        self.eval_interval = 10       # evaluer policy hver N episode
        self.num_eval_episodes = 5    # antall eval-episoder

        # Beregn nettverksstørrelser
        self._compute_network_sizes()

    def _compute_network_sizes(self):
        """Beregn input/output-størrelser for nettverkene basert på spillparametre."""
        # State: [agent_row, agent_col, delta_row, delta_col, manhattan, 
        #         at_top, at_bottom, at_left, at_right] = 9
        self.game_state_size = 9

        nnr_input = (self.lookback_q + 1) * self.game_state_size
        self.nnr_layers = [nnr_input, 128, 64, self.abstract_state_size]
        self.nnr_activations = ['relu', 'relu']  # ReLU er ofte bedre for dypere nettverk

        nnd_input = self.abstract_state_size + self.num_actions
        nnd_output = self.abstract_state_size + 1
        self.nnd_layers = [nnd_input, 128, 64, nnd_output]
        self.nnd_activations = ['relu', 'relu']

        nnp_input = self.abstract_state_size
        nnp_output = self.num_actions + 1
        self.nnp_layers = [nnp_input, 128, 64, nnp_output]
        self.nnp_activations = ['relu', 'relu']

    def set_small(self):
        """Sett små parametre for rask testing/debugging."""
        self.grid_size = 3
        self.num_episodes = 100
        self.steps_per_episode = 30
        self.num_mcts_searches = 20
        self.max_search_depth = 4
        self.abstract_state_size = 16
        self.lookback_q = 0
        self.rollout_w = 4
        self.minibatch_size = 16
        self.training_interval = 1
        self._compute_network_sizes()
        return self
