"""Game State Manager for MuZero.

Denne modulen inneholder GameStateManager som wrapper GridCollect-spillet
og gir et enhetlig grensesnitt for RL-systemet. Håndterer tilstandskonvertering,
handlinger, og terminalsjekk.
"""

import numpy as np
import jax.numpy as jnp
from grid_collect import GridCollect


class GameStateManager:

    def __init__(self, config):
        self.config = config
        self.game = GridCollect(config.grid_size)

    def reset(self):
        """
        Start nytt spill.
        """
        return self.game.reset()

    def get_legal_actions(self, state):
        """
        Returner lovlige handlinger.
        """
        return self.game.get_legal_actions(state)

    def step(self, state, action):
        """
        Utfør handling og returner resultat.
        """
        return self.game.step(state, action)

    def is_terminal(self, state):
        """
        Sjekk om tilstanden er terminal.
        """
        return self.game.is_terminal(state)

    def state_to_array(self, state):
        """
        Konverter tilstand til flat array for nevrale nettverk.
        """
        return self.game.state_to_array(state)

    def blank_state_array(self):
        """
        Returner en tom tilstand som flat array.
        Brukes for padding ved lookback når det ikke er nok historikk.
        """
        return np.zeros(self.config.game_state_size, dtype=np.float32)

    def render(self, state):
        self.game.render(state)
