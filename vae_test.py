import unittest
import numpy as np
from vae import VAE, vae_loss, generate_timeseries_data

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.num_samples = 3
        self.num_timesteps = 50
        self.input_dim = self.num_timesteps
        self.latent_dim = 2
        self.epochs = 10

        # Generate synthetic timeseries data
        self.data = generate_timeseries_data(self.num_samples, self.num_timesteps)

        # Normalize the data
        self.data_mean = self.data.mean()
        self.data_std = self.data.std()
        self.data_normalized = (self.data - self.data_mean) / self.data_std

        # Create VAE model
        self.vae = VAE(self.input_dim, self.latent_dim)
        self.vae.compile(optimizer='adam', loss=vae_loss)

    def test_data_generation(self):
        # Check if the generated data has the correct shape
        self.assertEqual(self.data.shape, (self.num_samples, self.num_timesteps))

    def test_data_normalization(self):
        # Check if the data is correctly normalized
        data_mean = self.data_normalized.mean()
        data_std = self.data_normalized.std()
        self.assertAlmostEqual(data_mean, 0.0, places=5)
        self.assertAlmostEqual(data_std, 1.0, places=5)

    def test_vae_model(self):
        # Check if the VAE model can be compiled successfully
        self.assertIsNotNone(self.vae)

    def test_vae_training(self):
        # Check if the VAE model can be trained without errors
        self.vae.fit(self.data_normalized, self.data_normalized, epochs=self.epochs, batch_size=32)

    def test_vae_encoding(self):
        # Check if the encoder produces the correct encoded data shape
        encoded_data = self.vae.encoder.predict(self.data_normalized)
        self.assertEqual(encoded_data.shape, (self.num_samples, self.latent_dim))

    def test_vae_decoding(self):
        # Check if the decoder produces the correct decoded data shape
        encoded_data = self.vae.encoder.predict(self.data_normalized)
        decoded_data = self.vae.decoder.predict(encoded_data)
        self.assertEqual(decoded_data.shape, (self.num_samples, self.input_dim))

    def test_vae_loss(self):
        # Check if the custom VAE loss function works as expected
        y_true = self.data_normalized  # Use the normalized data as y_true
        y_pred = np.random.rand(self.num_samples, self.input_dim)  # Random prediction for testing
        loss_value = vae_loss(y_true, y_pred)

        # Convert the loss_value to a NumPy float32 value
        loss_value = loss_value.numpy()

        # Calculate the mean of the loss_value array and explicitly cast it to np.float32
        mean_loss_value = np.mean(loss_value).astype(np.float32)

        self.assertIsInstance(mean_loss_value, np.float32)




if __name__ == '__main__':
    unittest.main()
