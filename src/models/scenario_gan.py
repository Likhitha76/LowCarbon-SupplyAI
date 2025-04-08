import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ScenarioGAN:
    def __init__(self, input_dim=100, output_dim=10):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(256, input_dim=self.input_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(512),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(1024),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Dense(self.output_dim, activation='tanh')
        ])
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Dense(512, input_dim=self.output_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(256),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = layers.Input(shape=(self.input_dim,))
        x = self.generator(gan_input)
        gan_output = self.discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan
    
    def train(self, real_data, epochs=1000, batch_size=32):
        real_data = np.array(real_data)
        batch_count = real_data.shape[0] // batch_size
        
        for epoch in range(epochs):
            for _ in range(batch_count):
                # Generate fake data
                noise = np.random.normal(0, 1, size=[batch_size, self.input_dim])
                fake_data = self.generator.predict(noise)
                
                # Train discriminator
                real_labels = np.ones((batch_size, 1))
                fake_labels = np.zeros((batch_size, 1))
                
                d_loss_real = self.discriminator.train_on_batch(real_data, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(fake_data, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train generator
                noise = np.random.normal(0, 1, size=[batch_size, self.input_dim])
                g_loss = self.gan.train_on_batch(noise, real_labels)
                
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
    
    def generate_scenarios(self, num_scenarios=10):
        noise = np.random.normal(0, 1, size=[num_scenarios, self.input_dim])
        return self.generator.predict(noise) 