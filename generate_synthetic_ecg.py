import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Load the trained generator model
generator_model_path = 'save_model/final_generator.h5'  # Path to your trained model
model = load_model(generator_model_path)

# 2. Generate synthetic ECG signals using random noise
num_signals = 100  # Number of synthetic signals to generate
latent_dim = 100  # Latent space dimensionality, same as the input size of the generator

# Generate random noise (latent space vectors) with a normal distribution
noise = np.random.normal(0, 1, (num_signals, latent_dim))

# Use the generator model to generate synthetic signals
generated_signals = model.predict(noise)

# 3. Visualize the generated signals
plt.figure(figsize=(10, 10))

# Plot the first 10 signals for example (adjust as needed)
for i in range(10):
    plt.subplot(5, 2, i + 1)  # 5 rows, 2 columns of subplots
    plt.plot(generated_signals[i].flatten())  # Flatten to 1D for better plotting
    plt.title(f"Generated Signal {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()
