import matplotlib.pyplot as plt
from config import Config
from pid_controller import PIDController
from bathtub_plant import BathtubPlant
from consys import CONSYS

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(results['mse_history'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Læringsprogresjon', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.plot(results['kp_history'], label='kp', linewidth=2, color='#A23B72')
    ax2.plot(results['ki_history'], label='ki', linewidth=2, color='#F18F01')
    ax2.plot(results['kd_history'], label='kd', linewidth=2, color='#C73E1D')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Parameter-verdi', fontsize=12)
    ax2.set_title('PID Parametere - Utvikling', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Lag konfigurasjon
config = Config()


# Lag controller
controller = PIDController(kp=0.001, ki=0.001, kd=0.0001)  # ← Dårligere start

# Lag plant
plant = BathtubPlant(
    A=config.bathtub_area,
    C=config.drain_area,
    H0=config.initial_height
)

# Lag kontrollsystem
consys = CONSYS(controller, plant, config)

# Tren
results = consys.train()
plot_results(results)

