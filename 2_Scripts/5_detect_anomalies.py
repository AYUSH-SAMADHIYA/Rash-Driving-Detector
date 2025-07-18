import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Configuration ---
ANOMALY_MODEL_PATH = 'Rash_Driving_Project/3_Models/anomaly_detector_model.h5'
DATA_PATH = 'Rash_Driving_Project/0_Data/processed/balanced_road_data.csv'
CHUNK_SIZE = 100  # Must match TIME_STEPS from training
BATCH_SIZE = 64   # How many sequences to process at a time

# --- NEW: Data Generator Class ---
class DataGenerator(Sequence):
    """
    Generates data for Keras models on-the-fly to save memory.
    """
    def __init__(self, df, feature_cols, batch_size, chunk_size):
        self.df = df
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.indexes = np.arange(0, len(df) - chunk_size, chunk_size)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_index:end_index]
        
        X = np.empty((self.batch_size, self.chunk_size, len(self.feature_cols)))
        
        for i, idx in enumerate(batch_indexes):
            X[i,] = self.df[self.feature_cols].values[idx : idx + self.chunk_size]
            
        return X

# --- Main Detection Logic ---
def detect_anomalies():
    print("Loading anomaly detector model...")
    try:
        model = load_model(ANOMALY_MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading data for detection...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at '{DATA_PATH}'.")
        return

    feature_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns].astype(np.float32))

    print("Setting up data generator...")
    generator = DataGenerator(df, feature_columns, BATCH_SIZE, CHUNK_SIZE)

    print("Calculating reconstruction errors using the generator...")
    # The model predicts on the generator batch by batch
    X_pred = model.predict(generator, verbose=1)

    # Now we need to get the original data (X) to compare against
    # We can use the generator again for this
    X_original = []
    for i in range(len(generator)):
        X_original.append(generator[i])
    X_original = np.concatenate(X_original)

    # Ensure X_pred and X_original have the same length
    min_len = min(len(X_pred), len(X_original))
    X_pred = X_pred[:min_len]
    X_original = X_original[:min_len]

    # Calculate the Mean Absolute Error (MAE) for each sequence
    mae_loss = np.mean(np.abs(X_pred - X_original), axis=(1, 2))
    
    plt.figure(figsize=(10, 5))
    plt.hist(mae_loss, bins=50)
    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("Number of Sequences")
    plt.title("Distribution of Reconstruction Errors")
    plt.show()
    
    ANOMALY_THRESHOLD = np.percentile(mae_loss, 96)
    print(f"\nAnomaly Threshold (95th percentile) set to: {ANOMALY_THRESHOLD:.4f}")

    print("\n--- Starting Anomaly Detection Simulation ---\n")
    for i in range(len(mae_loss)):
        error = mae_loss[i]
        
        print(f"Time Segment {i + 1}: Reconstruction Error = {error:.4f}")
        
        if error > ANOMALY_THRESHOLD:
            print(f"  -> \033[91mALERT: Rash Driving Anomaly Detected! (Error > Threshold)\033[0m")
        else:
            print("  -> Driving: Normal")
        print("-" * 60)

if __name__ == '__main__':
    detect_anomalies()
