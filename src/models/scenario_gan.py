import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict

class ScenarioGAN:
    def __init__(self, latent_dim: int = 100):
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
        self.trained = False
        
    def _build_generator(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=self.latent_dim),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(4, activation='tanh')  # 4 features: distance, emissions, traffic, weather
        ])
        return model
        
    def _build_discriminator(self) -> keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(256, input_dim=4),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
        
    def _build_gan(self) -> keras.Model:
        self.discriminator.trainable = False
        model = keras.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model
    
    def train(self, route_data: np.ndarray, epochs: int = 1000, batch_size: int = 32):
        """
        Train the GAN on route data
        """
        # Normalize the data to [-1, 1] range
        self.data_min = route_data.min(axis=0)
        self.data_max = route_data.max(axis=0)
        normalized_data = 2 * (route_data - self.data_min) / (self.data_max - self.data_min) - 1
        
        # Training loop
        for epoch in range(epochs):
            # Train discriminator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated_data = self.generator.predict(noise, verbose=0)
            
            idx = np.random.randint(0, normalized_data.shape[0], batch_size)
            real_data = normalized_data[idx]
            
            d_loss_real = self.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        
        self.trained = True
    
    def generate_scenarios(self, num_scenarios: int = 5) -> List[Dict]:
        """
        Generate future scenarios
        """
        if not self.trained:
            print("Warning: Model not trained, generating random scenarios")
            return self._generate_random_scenarios(num_scenarios)
        
        # Generate scenarios
        noise = np.random.normal(0, 1, (num_scenarios, self.latent_dim))
        generated_data = self.generator.predict(noise, verbose=0)
        
        # Denormalize the data
        generated_data = 0.5 * (generated_data + 1) * (self.data_max - self.data_min) + self.data_min
        
        # Convert to list of dictionaries
        scenarios = []
        for i in range(num_scenarios):
            scenario = {
                'distance': float(generated_data[i, 0]),
                'emissions': float(generated_data[i, 1]),
                'traffic_factor': float(generated_data[i, 2]),
                'weather_factor': float(generated_data[i, 3])
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_random_scenarios(self, num_scenarios: int = 5) -> List[Dict]:
        """
        Generate random scenarios when model is not trained
        """
        scenarios = []
        for _ in range(num_scenarios):
            scenario = {
                'distance': np.random.uniform(100, 500),  # km
                'emissions': np.random.uniform(50, 200),  # kg CO2
                'traffic_factor': np.random.uniform(0.8, 1.5),
                'weather_factor': np.random.uniform(0.9, 1.3)
            }
            scenarios.append(scenario)
        return scenarios 