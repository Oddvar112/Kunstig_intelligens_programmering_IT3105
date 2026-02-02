class Config:    
    def __init__(self):
        self.num_epochs = 100
        self.num_timesteps = 10
        self.learning_rate = 0.0001   # 0.01 funker bra for cournot 0.0001 for inflationn NN er nice
        self.noise_range = (-0.01, 0.01) # virker mye høyere enn bare kravet som er 0.01
        
        self.plant_type = "inflation"  # "bathtub" eller "cournot eller inflation"
        
        self.bathtub_area = 10
        self.drain_area = 0.3
        self.initial_height = 5
        
        self.pmax = 4              # Maximum price
        self.cm = 0.1                # Marginal cost
        self.target_profit = 2     # Target profit per timestep
        self.q1_init = 0.5           # Initial production (oss)
        self.q2_init = 0.5           # Initial production (rival)
        
        self.r0=0.04 # Current interest rate
        self.pi0=0.031 # inflation 2025
        self.target_pi=0.02 # Norges bank target
        self.alpha=0.5

        self.pid_kp_init = 0
        self.pid_ki_init = 0
        self.pid_kd_init = 0
        
        self.nn_layers = [3, 8, 8, 1]  
        self.nn_activation = 'tanh'     # 'tanh', 'sigmoid', eller 'relu'
        
        self.weight_init_range = (-0.5, 0.5)
        self.bias_init_range = (0.0, 0.0) # alle bias = 0