# Variational Autoencoder (VAE)

This repository contains an implementation of a Variational Autoencoder (VAE) for timeseries data. VAEs are a type of generative model that can learn to encode data into a lower-dimensional representation (latent space) and then decode it back to reconstruct the original data. This is achieved by combining an encoder neural network, which maps the input data to the latent space, and a decoder neural network, which reconstructs the data from the latent space.

## Understanding Autoencoders

For a detailed explanation of autoencoders and their various types, including VAEs, I recommend reading the Medium article ["Understanding Autoencoders"](https://medium.com/datadriveninvestor/understanding-autoencoders-8e228eb96cec). The article provides a comprehensive overview of autoencoders, their architecture, and their applications.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow
- Keras
- scikit-learn

You can install the required Python packages using the following command:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

## Usage

To run the VAE model on synthetic timeseries data, you can execute the `vae.py` script with command-line arguments:

```bash
python vae.py --num_samples 3 --num_timesteps 50 --latent_dim 2 --epochs 300 --batch_size 32
```

The command-line arguments allow you to control the number of samples, timesteps, latent space dimension, training epochs, and batch size for the VAE model.

## File Structure

The repository is organized as follows:

- `vae.py`: Python script containing the VAE model implementation and the main program.
- `README.md`: This file, providing information about the repository.
- `medium_article.md`: Markdown file containing the content of the Medium article "Understanding Autoencoders".

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

The VAE implementation in this repository is created by Ryan Martin.
