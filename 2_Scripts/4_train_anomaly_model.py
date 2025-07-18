import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# --- Configuration ---
PROCESSED_DATA_PATH = 'Rash_Driving_Project/0_Data/processed/balanced_road_data.csv'
ANOMALY_MODEL_SAVE_PATH = 'Rash_Driving_Project/3_Models/anomaly_detector_model.h5'

# LSTM Hyperparameters
TIME_STEPS = 100  # Must match the sequence length
STEP = 20
EPOCHS = 20       # Autoencoders can benefit from a few more epochs
BATCH_SIZE = 64

# --- Main Script ---
def train_anomaly_detector():
    """
    Loads the processed data and trains an LSTM Autoencoder to learn the
    patterns of normal driving.
    """
    print("Loading processed data for anomaly detection training...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at '{PROCESSED_DATA_PATH}'.")
        return

    feature_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Use float32 for memory efficiency
    for col in feature_columns:
        df[col] = df[col].astype(np.float32)

    print("Scaling features...")
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # --- Create Sequences (we only need X, no labels 'y' needed) ---
    print(f"Creating sequences with TIME_STEPS={TIME_STEPS}...")
    
    scaled_features = df[feature_columns].values
    sequences = []
    for i in range(0, len(df) - TIME_STEPS, STEP):
        sequences.append(scaled_features[i : i + TIME_STEPS])

    X = np.array(sequences)
    print(f"Created {X.shape[0]} sequences for training.")

    # Split data into training and validation sets
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # --- Build LSTM Autoencoder Model ---
    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    
    # Encoder
    inputs = Input(shape=(timesteps, n_features))
    encoding_dim = 16 # A smaller dimension to force the model to learn a compressed representation
    encoder = LSTM(encoding_dim)(inputs)

    # Decoder
    decoder = RepeatVector(timesteps)(encoder)
    decoder = LSTM(n_features, return_sequences=True)(decoder)
    
    autoencoder = Model(inputs=inputs, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mae') # Mean Absolute Error is a good loss for this
    
    print("\n--- Autoencoder Model Summary ---")
    autoencoder.summary()

    # --- Train the Model ---
    print("\n--- Starting Autoencoder Training ---")
    history = autoencoder.fit(
        X_train, X_train, # Note: The model learns to reconstruct itself (X -> X)
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, X_val),
        verbose=1
    )

    # --- Save the Model ---
    print(f"\nSaving anomaly detector model to '{ANOMALY_MODEL_SAVE_PATH}'...")
    autoencoder.save(ANOMALY_MODEL_SAVE_PATH)
    print("✅ Anomaly detector training complete!")

if __name__ == '__main__':
    train_anomaly_detector()
