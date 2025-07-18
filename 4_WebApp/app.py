from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import joblib

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Configuration ---
CHUNK_SIZE = 100
BATCH_SIZE = 64 # Process in batches to save memory
ANOMALY_MODEL_PATH = '3_Models/anomaly_detector_model.h5'
ROAD_MODEL_PATH = '3_Models/road_type_classifier.h5'
SCALER_PATH = '3_Models/scaler.pkl'
ROAD_TYPE_MAP = {0: 'Bitumin', 1: 'Block', 2: 'Concrete', 3: 'Kanker'}

# --- Load Models and Scaler Once at Startup ---
print("Loading models and scaler...")
anomaly_model = load_model(ANOMALY_MODEL_PATH, compile=False)
road_model = load_model(ROAD_MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("✅ Models loaded successfully!")

# --- NEW: Data Generator Class for Flask ---
class DataGenerator(Sequence):
    """Generates data for Keras models on-the-fly to save memory."""
    def __init__(self, df, feature_cols, batch_size, chunk_size):
        self.df = df
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.indexes = np.arange(0, len(df) - chunk_size + 1, chunk_size)

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_idx:end_idx]
        
        X = np.empty((len(batch_indexes), self.chunk_size, len(self.feature_cols)), dtype=np.float32)
        for i, idx in enumerate(batch_indexes):
            X[i,] = self.df[self.feature_cols].values[idx : idx + self.chunk_size]
        return X

# --- Define Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            df = pd.read_csv(file)
            feature_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

            df_scaled = df.copy()
            df_scaled[feature_columns] = scaler.transform(df[feature_columns].astype(np.float32))
            
            # Use the Data Generator
            generator = DataGenerator(df_scaled, feature_columns, BATCH_SIZE, CHUNK_SIZE)

            if len(generator) == 0:
                return jsonify({"error": "File is too short to analyze"}), 400

            # --- Perform Predictions using the generator ---
            X_pred = anomaly_model.predict(generator)
            road_preds = road_model.predict(generator)
            
            # Reconstruct original data from generator to calculate error
            X_original = np.concatenate([generator[i] for i in range(len(generator))])
            
            min_len = min(len(X_pred), len(X_original))
            mae_loss = np.mean(np.abs(X_pred[:min_len] - X_original[:min_len]), axis=(1, 2))
            road_classes = np.argmax(road_preds[:min_len], axis=1)

            # --- Format Results ---
            results = []
            for i in range(len(mae_loss)):
                results.append({
                    "segment": i + 1,
                    "road_type": ROAD_TYPE_MAP.get(road_classes[i], "Unknown"),
                    "error": float(mae_loss[i])
                })
            
            # Include raw data for plotting
            raw_data_for_plotting = df.iloc[:len(results)*CHUNK_SIZE]
            response_data = {
                "results": results,
                "sensor_data": {
                    "acc_x": list(raw_data_for_plotting['acc_x']),
                    "gyro_z": list(raw_data_for_plotting['gyro_z'])
                }
            }
            
            return jsonify(response_data)

        except Exception as e:
            # Provide a more specific error message for debugging
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
