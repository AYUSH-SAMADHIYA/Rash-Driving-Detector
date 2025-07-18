import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib # <-- Import joblib

# --- Configuration ---
PROCESSED_DATA_PATH = 'Rash_Driving_Project/0_Data/processed/balanced_road_data.csv'
MODEL_SAVE_PATH = 'Rash_Driving_Project/3_Models/road_type_classifier.h5'
SCALER_SAVE_PATH = 'Rash_Driving_Project/3_Models/scaler.pkl' # <-- Path to save the scaler

# LSTM Hyperparameters
TIME_STEPS = 100
STEP = 20
EPOCHS = 15
BATCH_SIZE = 64

# --- Main Script ---
def train_model():
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    feature_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # --- 1. Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns].astype(np.float32))

    # --- NEW: Save the scaler ---
    print(f"Saving scaler to '{SCALER_SAVE_PATH}'...")
    joblib.dump(scaler, SCALER_SAVE_PATH)
    
    # --- The rest of the script is the same ---
    print(f"Creating sequences...")
    scaled_features = df[feature_columns].values
    labels = df['label'].values
    X, y = [], []
    for i in range(0, len(df) - TIME_STEPS, STEP):
        features_chunk = scaled_features[i : i + TIME_STEPS]
        label = stats.mode(labels[i : i + TIME_STEPS], keepdims=True)[0][0]
        X.append(features_chunk)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    num_classes = len(np.unique(y))
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = Sequential([ LSTM(32, input_shape=input_shape, return_sequences=True), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(16, activation='relu'), Dense(num_classes, activation='softmax') ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("\n--- Starting Model Training ---")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)
    print("\n--- Evaluating Final Model ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    print(f"\nSaving model to '{MODEL_SAVE_PATH}'...")
    model.save(MODEL_SAVE_PATH)
    print("✅ Model and scaler saved!")

if __name__ == '__main__':
    train_model()
