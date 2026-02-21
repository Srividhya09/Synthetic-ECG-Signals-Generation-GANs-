import numpy as np
import pickle
from utils.data_util import *
from GAN_model import *

if __name__ == "__main__":
    print("Loading data...")

    # Load the AF dataset
    X_train = pickle.load(open("X_train_af.pkl", "rb"))
    y = pickle.load(open("y_af.pkl", "rb"))

    # Initialize the DataLoader and filter only AF signals
    dataloader = DataLoader()
    X_train = dataloader.pick_type_only(X_train, y, 1)  # Pick AF ECG only (label = 1)

    if X_train.size == 0:
        raise ValueError("No AF signals found in the dataset. Check the label filtering.")

    # GAN Training Parameters
    EPOCHS = 10000
    LATENT_SIZE = 100
    SAVE_INTRIVAL = 100
    SAVE_MODEL_INTERVAL = 1000
    BATCH_SIZE = 128
    INPUT_SHAPE = (180, 1)  # Shape for AF dataset
    RANDOM_SINE = False
    SCALE = 2
    MINIBATCH = True  # Use minibatch discrimination to avoid mode collapse
    SAVE_MODEL = True
    SAVE_REPORT = True
    GEN_VERSION = 0  # 0 = Default generator; 1-5 = Alternate generator versions

    # Initialize the DCGAN model
    dcgan = DCGAN(
        INPUT_SHAPE, LATENT_SIZE, 
        random_sine=RANDOM_SINE, 
        scale=SCALE, 
        minibatch=MINIBATCH, 
        gen_version=GEN_VERSION
    )

    # Preprocess the training data
    X_train = dcgan.specify_range(X_train, -2, 2) / 2  # Limit range to [-2, 2], then scale
    X_train = X_train.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])  # Reshape for input shape

    print(f"Training with {X_train.shape[0]} signals...")

    # Train the GAN
    dcgan.train(
        EPOCHS, X_train, BATCH_SIZE, 
        SAVE_INTRIVAL, save=SAVE_MODEL, 
        save_model_interval=SAVE_MODEL_INTERVAL, 
        save_report=SAVE_REPORT
    )

    # Save the trained model after training completes
    if SAVE_MODEL:
        print("Saving trained model...")
        dcgan.save_model("trained_dcgan_model.h5")  # Save the final model to file

    print("Training Complete!")
