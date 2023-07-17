import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from tensorflow import keras
from sklearn.manifold import TSNE
from keras import layers, models, backend
import argparse


def generate_timeseries_data(num_samples, num_timesteps):
    """
    Generate synthetic timeseries data.

    Parameters:
        num_samples (int): Number of samples in the dataset.
        num_timesteps (int): Number of timesteps in each sample.

    Returns:
        np.ndarray: Generated timeseries data with shape (num_samples, num_timesteps).
    """
    # Your custom data generation code here
    # For simplicity, let's generate random data
    return np.random.rand(num_samples, num_timesteps)


class VAE(models.Model):
    """
    Variational Autoencoder (VAE) model.

    Parameters:
        input_dim (int): Dimension of input timeseries data.
        latent_dim (int): Dimension of the latent space.

    Attributes:
        input_dim (int): Dimension of input timeseries data.
        latent_dim (int): Dimension of the latent space.
        encoder (tf.keras.models.Model): Encoder model.
        decoder (tf.keras.models.Model): Decoder model.
    """

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_input = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(128, activation='relu')(encoder_input)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        encoder_output = layers.Lambda(self.sample_z, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        return models.Model(encoder_input, encoder_output)

    def build_decoder(self):
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation='relu')(decoder_input)
        decoder_output = layers.Dense(self.input_dim, activation='sigmoid')(x)
        return models.Model(decoder_input, decoder_output)

    def sample_z(self, args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = backend.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions


def vae_loss(y_true, y_pred):
    """
    Define the loss function for the VAE model.

    Parameters:
        y_true (tf.Tensor): Ground truth values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Total loss (reconstruction loss + KL divergence).
    """
    z_mean, z_log_var = tf.split(y_pred, num_or_size_splits=2, axis=1)
    recon_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return recon_loss + kl_loss


def main(num_samples, num_timesteps, latent_dim, epochs, batch_size):
    # Generate synthetic timeseries data
    data = generate_timeseries_data(num_samples, num_timesteps)

    # Normalize the data
    data_mean = data.mean()
    data_std = data.std()
    data_normalized = (data - data_mean) / data_std

    # Create VAE model
    vae = VAE(num_timesteps, latent_dim)
    vae.compile(optimizer='adam', loss=vae_loss)

    # Train the VAE
    vae.fit(data_normalized, data_normalized, epochs=epochs, batch_size=batch_size)

    # Reduce dimensionality of timeseries using the trained encoder
    encoded_data = vae.encoder.predict(data_normalized)

    # Now, 'encoded_data' contains the lower-dimensional representations of the timeseries data

    plt.figure(figsize=(12, 6))

    # Plot the original timeseries data
    plt.subplot(4, 2, 1)
    plt.title("Original Timeseries Data")
    for i in range(num_samples):
        plt.plot(data[i], alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Plot the encoded data in 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    encoded_data_tsne = tsne.fit_transform(encoded_data)

    plt.subplot(4, 2, 2)
    plt.title("Encoded Data (t-SNE)")
    for i in range(num_samples):
        plt.scatter(encoded_data_tsne[i, 0], encoded_data_tsne[i, 1], alpha=0.5, label=f"Sample {i}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()

    # Plot the encoded data in 2D
    plt.subplot(4, 2, 3)
    plt.title("Encoded Data (2D)")
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], alpha=0.5)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    # Plot the original timeseries data after applying t-SNE
    decoded_data_tsne = vae.decoder.predict(encoded_data_tsne)
    plt.subplot(4, 2, 4)
    plt.title("Reconstructed Timeseries Data (t-SNE)")
    for i in range(num_samples):
        plt.plot(decoded_data_tsne[i], alpha=0.5)
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder for Timeseries Data")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples in the dataset")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of timesteps in each sample")
    parser.add_argument("--latent_dim", type=int, default=2, help="Dimension of the latent space")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()
    main(args.num_samples, args.num_timesteps, args.latent_dim, args.epochs, args.batch_size)



