
class Config:    
    def __init__(self):
        self.num_epochs = 100
        self.num_timesteps = 10
        self.learning_rate = 0.1
        self.noise_range = (-0.03, 0.03)
        
        self.bathtub_area = 10
        self.drain_area = 0.3
        self.initial_height = 5
               
        # Classic PID start-verdier
        self.pid_kp_init = 1
        self.pid_ki_init = 1
        self.pid_kd_init = 1
        
        self.nn_layers = [3, 8, 8, 1]  
        self.nn_activation = 'tanh'     # 'tanh', 'sigmoid', eller 'relu'