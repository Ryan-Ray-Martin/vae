import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.manifold import TSNE
from keras import layers, models
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
    return np.random.rand(num_samples, num_timesteps)

class VAE(models.Model):
    """
    Variational Autoencoder (VAE) model.

    Parameters:
        input_dim (int): Dimension of input timeseries data.
        latent_dim (int): Dimension of the latent space.
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
        return models.Model(encoder_input, [z_mean, z_log_var, encoder_output])  # Ensure to return all three outputs

    def build_decoder(self):
        decoder_input = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation='relu')(decoder_input)
        decoder_output = layers.Dense(self.input_dim, activation='sigmoid')(x)
        return models.Model(decoder_input, decoder_output)

    def sample_z(self, args):
        z_mean, z_log_var = args
        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)  # Correctly unpack all three outputs
        reconstructions = self.decoder(z)
        return reconstructions, z_mean, z_log_var


def vae_loss(y_true, y_pred, z_mean, z_log_var):
    """
    Define the loss function for the VAE model.

    Parameters:
        y_true (tf.Tensor): Ground truth values.
        y_pred (tf.Tensor): Predicted values (reconstructed data).
        z_mean (tf.Tensor): Mean of the latent variable.
        z_log_var (tf.Tensor): Log variance of the latent variable.

    Returns:
        tf.Tensor: Total loss (reconstruction loss + KL divergence).
    """
    mse_loss = tf.keras.losses.MeanSquaredError()
    recon_loss = mse_loss(y_true, y_pred)
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
    vae.compile(optimizer='adam')  # Remove the custom loss from here

    # Custom training loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            reconstructions, z_mean, z_log_var = vae(data_normalized)
            loss = vae_loss(data_normalized, reconstructions, z_mean, z_log_var)  # Pass the outputs to the loss function
        
        grads = tape.gradient(loss, vae.trainable_variables)
        vae.optimizer.apply_gradients(zip(grads, vae.trainable_variables))

        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')  # Print the loss for each epoch

    # Reduce dimensionality of timeseries using the trained encoder
    z_mean, z_log_var, encoded_data = vae.encoder.predict(data_normalized)  # Unpack the outputs to get the encoded data

    # Set the style of the plots
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 10))

    # Plot the original timeseries data
    plt.subplot(4, 2, 1)
    plt.title("Original Timeseries Data", fontsize=16)
    for i in range(num_samples):
        plt.plot(data[i], alpha=0.5, label=f"Sample {i+1}")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # Plot the encoded data in 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    encoded_data_tsne = tsne.fit_transform(encoded_data)

    plt.subplot(4, 2, 2)
    plt.title("Encoded Data (t-SNE)", fontsize=16)
    plt.scatter(encoded_data_tsne[:, 0], encoded_data_tsne[:, 1], 
                alpha=0.7, s=100, c=range(num_samples), cmap='viridis')
    plt.colorbar(label='Sample Index')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True)

    # Plot the encoded data in 2D
    plt.subplot(4, 2, 3)
    plt.title("Latent Space Representation", fontsize=16)
    plt.scatter(encoded_data[:, 0], encoded_data[:, 1], 
                alpha=0.7, s=100, c=range(num_samples), cmap='viridis')
    plt.colorbar(label='Sample Index')
    plt.xlabel("Latent Dimension 1", fontsize=12)
    plt.ylabel("Latent Dimension 2", fontsize=12)
    plt.grid(True)

    # Plot the original timeseries data after applying t-SNE
    decoded_data_tsne = vae.decoder.predict(encoded_data_tsne)
    plt.subplot(4, 2, 4)
    plt.title("Reconstructed Timeseries Data from t-SNE", fontsize=16)
    for i in range(num_samples):
        plt.plot(decoded_data_tsne[i], alpha=0.5, label=f"Sample {i+1}")
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

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




