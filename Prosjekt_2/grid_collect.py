"""GridCollect spillmiljø for MuZero.

Denne modulen implementerer et enkelt 2D grid-basert spill hvor en agent
må navigere til mål-posisjoner. Brukes som testmiljø for MuZero-implementasjonen.

Spillet:
- Agent (A) og mål (X) plasseres på et NxN rutenett
- Agent kan bevege seg i 4 retninger (opp, høyre, ned, venstre)
- Positive reward ved å nå målet, liten negativ reward ellers
- Målet respawner på ny posisjon når det samles
"""

import numpy as np




class GridCollectState:

    def __init__(self, agent_pos, target_pos, grid_size):
        self.agent_pos = agent_pos # (row, col)
        self.target_pos = target_pos # (row, col)
        self.grid_size = grid_size

class GridCollect:

    # Action-konstanter
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_actions = 4

    def reset(self):
        """
        Start nytt spill med tilfeldige posisjoner.
        """
        agent_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )
        target_pos = agent_pos
        while target_pos == agent_pos:
            target_pos = (
                np.random.randint(self.grid_size),
                np.random.randint(self.grid_size)
            )
        return GridCollectState(agent_pos, target_pos, self.grid_size)

    def step(self, state, action):
        """
        Utfør handling og oppdater tilstand.
        """
        row, col = state.agent_pos

        if action == self.UP:
            row = max(0, row - 1)
        elif action == self.DOWN:
            row = min(self.grid_size - 1, row + 1)
        elif action == self.LEFT:
            col = max(0, col - 1)
        elif action == self.RIGHT:
            col = min(self.grid_size - 1, col + 1)

        new_agent_pos = (row, col)

        if new_agent_pos == state.target_pos:
            reward = 1.0  # Høyere reward for å nå mål
            # Respawn mål på ny posisjon
            new_target = new_agent_pos
            while new_target == new_agent_pos:
                new_target = (
                    np.random.randint(self.grid_size),
                    np.random.randint(self.grid_size)
                )
            return GridCollectState(new_agent_pos, new_target, self.grid_size), reward
        else:
            return GridCollectState(new_agent_pos, state.target_pos, self.grid_size), -0.01

    def get_legal_actions(self, state):
        """
        Returner lovlige handlinger.
        """
        return [0, 1, 2, 3]

    def is_terminal(self, state):
        """
        Sjekk om spillet er ferdig.
        """
        return False

    def state_to_array(self, state):
        """
        Konverter til kompakt feature-array for nevrale nettverk.
        """
        norm = max(1, self.grid_size - 1)
        
        # Relativ posisjon til mål
        delta_row = (state.target_pos[0] - state.agent_pos[0]) / norm
        delta_col = (state.target_pos[1] - state.agent_pos[1]) / norm
        
        # Manhattan-avstand normalisert
        manhattan = (abs(state.target_pos[0] - state.agent_pos[0]) + 
                    abs(state.target_pos[1] - state.agent_pos[1])) / (2 * norm)
        
        # Vegg-indikatorer (1 hvis ved kant, 0 ellers)
        at_top = 1.0 if state.agent_pos[0] == 0 else 0.0
        at_bottom = 1.0 if state.agent_pos[0] == self.grid_size - 1 else 0.0
        at_left = 1.0 if state.agent_pos[1] == 0 else 0.0
        at_right = 1.0 if state.agent_pos[1] == self.grid_size - 1 else 0.0
        
        return np.array([
            state.agent_pos[0] / norm,
            state.agent_pos[1] / norm,
            delta_row,
            delta_col,
            manhattan,
            at_top,
            at_bottom,
            at_left,
            at_right
        ], dtype=np.float32)

    def render(self, state):
        """
        Vis spilltilstanden i konsollen.
        """
        border = '+' + '-' * self.grid_size + '+'
        print(border)
        for r in range(self.grid_size):
            row_str = '|'
            for c in range(self.grid_size):
                if (r, c) == state.agent_pos and (r, c) == state.target_pos:
                    row_str += '@'
                elif (r, c) == state.agent_pos:
                    row_str += 'A'
                elif (r, c) == state.target_pos:
                    row_str += 'X'
                else:
                    row_str += '.'
            row_str += '|'
            print(row_str)
        print(border)
