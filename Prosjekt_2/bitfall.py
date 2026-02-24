import numpy as np
import copy


class BitFallState:
    """Representerer tilstanden i BitFall-spillet."""

    def __init__(self, grid, receptors, receptor_offset):
        self.grid = grid.copy()                    # 2D array (height x width), 1=debris, 0=empty
        self.receptors = list(receptors)            # liste med segment-størrelser, f.eks. [3]
        self.receptor_offset = receptor_offset      # kolonne-offset for reseptorene

    def copy(self):
        return BitFallState(
            self.grid.copy(),
            list(self.receptors),
            self.receptor_offset
        )


class BitFall:
    """BitFall-spillsimulator.

    Debris (røde segmenter) faller nedover en rad per tidssteg.
    Reseptorer (blå segmenter) på bunnen kan flyttes venstre/høyre/stå stille.
    Scoring basert på sammenligning mellom nederste debris-rad og reseptorer.
    """

    # Handlinger
    LEFT = 0
    STAY = 1
    RIGHT = 2

    def __init__(self, width, height, initial_receptors):
        self.width = width
        self.height = height
        self.initial_receptors = list(initial_receptors)  # f.eks. [3]

    def reset(self):
        """Start nytt spill med tomt grid og tilfeldige debris."""
        grid = np.zeros((self.height, self.width), dtype=np.float32)
        # Fyll noen rader med tilfeldig debris
        for row in range(self.height):
            grid[row] = self._random_debris_row()

        # Reseptorer starter sentrert
        total_receptor_width = sum(self.initial_receptors)
        receptor_offset = (self.width - total_receptor_width) // 2

        return BitFallState(grid, list(self.initial_receptors), receptor_offset)

    def _random_debris_row(self):
        """Generer en tilfeldig rad med debris-segmenter."""
        row = np.zeros(self.width, dtype=np.float32)
        # Generer tilfeldige segmenter
        col = 0
        while col < self.width:
            if np.random.random() < 0.4:  # 40% sjanse for debris-start
                seg_len = np.random.randint(1, min(4, self.width - col + 1))
                row[col:col + seg_len] = 1.0
                col += seg_len
            else:
                col += 1
        return row

    def get_legal_actions(self, state):
        """Alle tre handlinger er alltid lovlige."""
        return [self.LEFT, self.STAY, self.RIGHT]

    def is_terminal(self, state):
        """BitFall har ingen sluttilstand."""
        return False

    def step(self, state, action):
        """Utfør en handling og returner ny tilstand + reward.

        1. Flytt reseptorer basert på handling
        2. Beregn scoring mellom nederste debris-rad og reseptorer
        3. Fjern nederste debris-rad
        4. Flytt alle debris en rad ned
        5. Legg til ny tilfeldig debris-rad på toppen
        """
        new_state = state.copy()

        # 1. Flytt reseptorer
        if action == self.LEFT:
            new_state.receptor_offset -= 1
        elif action == self.RIGHT:
            new_state.receptor_offset += 1
        # STAY: ingen endring

        # Håndter wrapping/splitting av reseptorer
        new_state = self._handle_receptor_wrapping(new_state)

        # 2. Beregn scoring med nederste rad
        reward = self._compute_reward(new_state)

        # 3-5. Oppdater grid: flytt debris ned, legg til ny rad på toppen
        new_grid = np.zeros_like(new_state.grid)
        # Flytt alle rader en ned (rad 0 er toppen)
        new_grid[1:] = new_state.grid[:-1]
        # Ny tilfeldig rad på toppen
        new_grid[0] = self._random_debris_row()
        new_state.grid = new_grid

        return new_state, float(reward)

    def _handle_receptor_wrapping(self, state):
        """Håndter reseptor-wrapping når de går forbi kantene."""
        total_width = sum(state.receptors)

        # Sjekk venstre kant
        if state.receptor_offset < 0:
            # Split: del som går forbi venstre kant wrapper til høyre
            overflow = -state.receptor_offset
            state.receptor_offset = 0
            # Juster reseptorer - forenklet: bare clamp offset
            if overflow >= total_width:
                state.receptor_offset = self.width - total_width

        # Sjekk høyre kant
        right_edge = state.receptor_offset + total_width
        if right_edge > self.width:
            overflow = right_edge - self.width
            # Clamp til høyre kant
            state.receptor_offset = self.width - total_width
            if state.receptor_offset < 0:
                state.receptor_offset = 0

        return state

    def _compute_reward(self, state):
        """Beregn reward basert på sammenligning av nederste debris-rad og reseptorer.

        Regler:
        - Reseptor dekker debris helt med overskudd på en/begge sider -> +poeng (debris-størrelse)
        - Debris dekker reseptor helt med overskudd -> -poeng (reseptor-størrelse)
        - Ellers: 0 poeng
        """
        bottom_row = state.grid[-1]  # nederste rad med debris
        reward = 0.0

        # Bygg reseptor-rad
        receptor_row = np.zeros(self.width, dtype=np.float32)
        pos = state.receptor_offset
        for seg_len in state.receptors:
            end = min(pos + seg_len, self.width)
            if pos >= 0 and pos < self.width:
                receptor_row[max(0, pos):end] = 1.0
            pos += seg_len

        # Finn sammenhengende segmenter i debris og reseptorer
        debris_segments = self._find_segments(bottom_row)
        receptor_segments = self._find_segments(receptor_row)

        # Sammenlign overlappende segmenter
        for d_start, d_len in debris_segments:
            d_end = d_start + d_len
            for r_start, r_len in receptor_segments:
                r_end = r_start + r_len

                # Sjekk overlapping
                overlap_start = max(d_start, r_start)
                overlap_end = min(d_end, r_end)

                if overlap_start < overlap_end:
                    # Det er overlapping - sjekk hvem som dekker hvem
                    if r_start <= d_start and r_end >= d_end and r_len > d_len:
                        # Reseptor dekker debris helt med overskudd -> +poeng
                        reward += d_len
                    elif d_start <= r_start and d_end >= r_end and d_len > r_len:
                        # Debris dekker reseptor helt med overskudd -> -poeng
                        reward -= r_len

        return reward

    def _find_segments(self, row):
        """Finn sammenhengende segmenter (start, lengde) i en rad."""
        segments = []
        i = 0
        while i < len(row):
            if row[i] > 0.5:
                start = i
                while i < len(row) and row[i] > 0.5:
                    i += 1
                segments.append((start, i - start))
            else:
                i += 1
        return segments

    def state_to_array(self, state):
        """Konverter spilltilstand til flat array for nevrale nettverk.

        Returnerer: flat array med grid (height*width) + receptor row (width)
        """
        # Bygg reseptor-rad
        receptor_row = np.zeros(self.width, dtype=np.float32)
        pos = state.receptor_offset
        for seg_len in state.receptors:
            end = min(pos + seg_len, self.width)
            if pos >= 0 and pos < self.width:
                receptor_row[max(0, pos):end] = 1.0
            pos += seg_len

        # Kombiner grid og reseptor-rad
        flat = np.concatenate([state.grid.flatten(), receptor_row])
        return flat

    def render(self, state):
        """Vis spilltilstanden i konsollen."""
        print("+" + "-" * self.width + "+")
        for row in range(self.height):
            line = "|"
            for col in range(self.width):
                if state.grid[row, col] > 0.5:
                    line += "#"
                else:
                    line += " "
            line += "|"
            print(line)

        # Vis reseptorer
        receptor_line = "|"
        pos = state.receptor_offset
        for col in range(self.width):
            is_receptor = False
            p = state.receptor_offset
            for seg_len in state.receptors:
                if p <= col < p + seg_len:
                    is_receptor = True
                    break
                p += seg_len
            receptor_line += "=" if is_receptor else " "
        receptor_line += "|"
        print(receptor_line)
        print("+" + "-" * self.width + "+")
