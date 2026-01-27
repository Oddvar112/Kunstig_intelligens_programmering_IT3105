import matplotlib.pyplot as plt
from config import Config
from inflation_plant import inflation_plant
from pid_controller import PIDController
from nn_pid_controller import NeuralPIDController
from bathtub_plant import BathtubPlant
from cournot_plant import CournotPlant
from consys import CONSYS

def plot_results(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(results['mse_history'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Læring', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    if 'kp_history' in results:
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

# konfigurasjon data objekt 
config = Config()

if config.plant_type == "bathtub":
    plant = BathtubPlant(
        A=config.bathtub_area,
        C=config.drain_area,
        H0=config.initial_height
    )
elif config.plant_type == "cournot":
    plant = CournotPlant(
        pmax=config.pmax,
        cm=config.cm,
        target_profit=config.target_profit,
        q1_init=config.q1_init,
        q2_init=config.q2_init
    )
elif config.plant_type == "inflation":
    print("inflation chosen")
    plant = inflation_plant(
        r0=config.r0,
        pi0=config.pi0,
        target_pi=config.target_pi,
        alpha=config.alpha
    )
    
else:
    raise ValueError(f"Unknown plant type: {config.plant_type}. Use 'bathtub' or 'cournot'")



# Velg controller type
print("\nVelg controller type:")
print("1. Classic PID")
print("2. Neural Network")
choice = input("Skriv 1 eller 2: ")

if choice == '1':
    controller = PIDController(
        kp=config.pid_kp_init, 
        ki=config.pid_ki_init, 
        kd=config.pid_kd_init
    )
elif choice == '2':
    controller = NeuralPIDController(
        layers=config.nn_layers, 
        activation=config.nn_activation,
        weight_init_range=config.weight_init_range,
        bias_init_range=config.bias_init_range
    )
else:
    print("Ugyldig valg, bruker Classic PID")
    controller = PIDController(
        kp=config.pid_kp_init, 
        ki=config.pid_ki_init, 
        kd=config.pid_kd_init
    )

consys = CONSYS(controller, plant, config)
results = consys.train()

plot_results(results)