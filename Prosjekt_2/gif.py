import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

steps = [
    {"agent": (3,3), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (3,2), "target": (0,0), "action": "RIGHT", "reward": 0},
    {"agent": (3,3), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (3,2), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (3,1), "target": (0,0), "action": "UP",    "reward": 0},
    {"agent": (2,1), "target": (0,0), "action": "UP",    "reward": 0},
    {"agent": (1,1), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (0,0), "action": "UP",    "reward": 0},
    {"agent": (0,0), "target": (2,1), "action": "DOWN",  "reward": 1.0},
    {"agent": (1,0), "target": (2,1), "action": "DOWN",  "reward": 0},
    {"agent": (2,0), "target": (2,1), "action": "RIGHT", "reward": 0},
    {"agent": (2,1), "target": (2,3), "action": "RIGHT", "reward": 1.0},
    {"agent": (2,2), "target": (2,3), "action": "RIGHT", "reward": 0},
    {"agent": (2,3), "target": (1,2), "action": "UP",    "reward": 1.0},
    {"agent": (1,3), "target": (1,2), "action": "UP",    "reward": 0},
    {"agent": (0,3), "target": (1,2), "action": "LEFT",  "reward": 0},
    {"agent": (0,2), "target": (1,2), "action": "DOWN",  "reward": 0},
    {"agent": (1,2), "target": (0,3), "action": "RIGHT", "reward": 1.0},
    {"agent": (1,3), "target": (0,3), "action": "UP",    "reward": 0},
    {"agent": (0,3), "target": (1,1), "action": "LEFT",  "reward": 1.0},
    {"agent": (0,2), "target": (1,1), "action": "LEFT",  "reward": 0},
    {"agent": (0,1), "target": (1,1), "action": "DOWN",  "reward": 0},
    {"agent": (1,1), "target": (1,3), "action": "RIGHT", "reward": 1.0},
    {"agent": (1,2), "target": (1,3), "action": "RIGHT", "reward": 0},
    {"agent": (1,3), "target": (2,2), "action": "LEFT",  "reward": 1.0},
    {"agent": (1,2), "target": (2,2), "action": "DOWN",  "reward": 0},
    {"agent": (2,2), "target": (3,0), "action": "LEFT",  "reward": 1.0},
    {"agent": (2,1), "target": (3,0), "action": "UP",    "reward": 0},
    {"agent": (1,1), "target": (3,0), "action": "DOWN",  "reward": 0},
    {"agent": (2,1), "target": (3,0), "action": "LEFT",  "reward": 0},
    {"agent": (2,0), "target": (3,0), "action": "DOWN",  "reward": 0},
    {"agent": (3,0), "target": (2,1), "action": "RIGHT", "reward": 1.0},
    {"agent": (3,1), "target": (2,1), "action": "UP",    "reward": 0},
    {"agent": (2,1), "target": (0,1), "action": "UP",    "reward": 1.0},
    {"agent": (1,1), "target": (0,1), "action": "UP",    "reward": 0},
    {"agent": (0,1), "target": (0,3), "action": "RIGHT", "reward": 1.0},
    {"agent": (0,2), "target": (0,3), "action": "RIGHT", "reward": 0},
    {"agent": (0,3), "target": (2,0), "action": "LEFT",  "reward": 1.0},
    {"agent": (0,2), "target": (2,0), "action": "LEFT",  "reward": 0},
    {"agent": (0,1), "target": (2,0), "action": "DOWN",  "reward": 0},
    {"agent": (1,1), "target": (2,0), "action": "LEFT",  "reward": 0},
    {"agent": (1,0), "target": (2,0), "action": "DOWN",  "reward": 0},
    {"agent": (2,0), "target": (0,2), "action": "UP",    "reward": 1.0},
    {"agent": (1,0), "target": (0,2), "action": "RIGHT", "reward": 0},
]

GRID = 4
cumulative = []
total = 0
for s in steps:
    total += s["reward"]
    cumulative.append(total)

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor('#2b2b2b')

def draw_frame(i):
    ax.clear()
    step = steps[i]
    agent = step["agent"]
    target = step["target"]

    ax.set_facecolor('#2b2b2b')
    ax.set_xlim(-0.1, GRID)
    ax.set_ylim(-0.1, GRID)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Steg {i}  |  Maal: {int(cumulative[i])}  |  Reward: {cumulative[i]:.1f}',
                 color='white', fontsize=12, pad=8)

    # Grid linjer
    for x in range(GRID + 1):
        ax.axvline(x, color='#555555', linewidth=1.0)
        ax.axhline(x, color='#555555', linewidth=1.0)

    # Celler (graa)
    for r in range(GRID):
        for c in range(GRID):
            ax.add_patch(patches.FancyBboxPatch(
                (c + 0.05, GRID - r - 1 + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor='#4a4a4a', edgecolor='none'))

    # Maal (gronn X)
    tr, tc = target
    ax.add_patch(patches.FancyBboxPatch(
        (tc + 0.05, GRID - tr - 1 + 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.05",
        facecolor='#2d6a2d', edgecolor='#4caf50', linewidth=2))
    ax.text(tc + 0.5, GRID - tr - 0.5, 'X',
            ha='center', va='center', fontsize=24,
            color='#4caf50', fontweight='bold')

    # Agent (blaa A)
    ar, ac = agent
    ax.add_patch(patches.FancyBboxPatch(
        (ac + 0.05, GRID - ar - 1 + 0.05), 0.9, 0.9,
        boxstyle="round,pad=0.05",
        facecolor='#1565c0', edgecolor='#42a5f5', linewidth=2))
    ax.text(ac + 0.5, GRID - ar - 0.5, 'A',
            ha='center', va='center', fontsize=22,
            color='white', fontweight='bold')

    # Flash ved innsamling
    if step["reward"] > 0:
        ax.add_patch(patches.Rectangle(
            (-0.1, -0.1), GRID + 0.2, GRID + 0.2,
            facecolor='#4caf50', alpha=0.18, zorder=10))
        ax.text(GRID/2, GRID/2, '+1',
                ha='center', va='center', fontsize=26,
                color='#4caf50', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#2b2b2b', alpha=0.85),
                zorder=11)

    plt.tight_layout(pad=0.5)

ani = animation.FuncAnimation(fig, draw_frame, frames=len(steps),
                               interval=400, repeat=True)

output = "muzero_demo.gif"
print(f"Lagrer {output} ...")
ani.save(output, writer='pillow', fps=2, dpi=150)
print(f"Ferdig! Lagret til {output}")
plt.close()