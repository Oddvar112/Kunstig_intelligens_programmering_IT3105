import numpy as np
import jax.numpy as jnp
from grid_collect import GridCollect


class GameStateManager:
    """Game State Manager - wrapper rundt GridCollect med tilleggsfunksjonalitet."""

    def __init__(self, config):
        self.config = config
        self.game = GridCollect(config.grid_size)

    def reset(self):
        """Start nytt spill."""
        return self.game.reset()

    def get_legal_actions(self, state):
        """Returner lovlige handlinger."""
        return self.game.get_legal_actions(state)

    def step(self, state, action):
        """Utfør handling og returner (ny_tilstand, reward)."""
        return self.game.step(state, action)

    def is_terminal(self, state):
        """Sjekk om tilstanden er terminal."""
        return self.game.is_terminal(state)

    def state_to_array(self, state):
        """Konverter tilstand til flat array for nevrale nettverk."""
        return self.game.state_to_array(state)

    def blank_state_array(self):
        """Returner en tom tilstand som flat array (for padding ved lookback)."""
        return np.zeros(self.config.game_state_size, dtype=np.float32)

    def render(self, state):
        """Vis tilstanden."""
        self.game.render(state)
