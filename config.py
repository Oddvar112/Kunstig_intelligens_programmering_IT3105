class Config:    
    def __init__(self):
        self.num_epochs = 100
        self.num_timesteps = 10
        self.learning_rate = 0.001
        
        self.noise_range = (-0.03, 0.03)  
        
        self.bathtub_area = 1.0
        self.drain_area = 0.02      
        self.initial_height = 1.0
        
