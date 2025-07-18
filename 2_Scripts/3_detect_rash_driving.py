import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
MODEL_PATH = 'Rash_Driving_Project/3_Models/road_type_classifier.h5'
DATA_PATH = 'Rash_Driving_Project/0_Data/processed/balanced_road_data.csv'
CHUNK_SIZE = 100 

# --- CALIBRATION: NEW, MORE LENIENT THRESHOLDS ---
# These values have been significantly increased to reduce false alerts.
# We are now looking for truly extreme events.
RASH_DRIVING_RULES = {
    # Road Type 0: 'Bitumin'
    0: {'max_accel': 8.0, 'min_accel': -8.0, 'max_turn': 30.0},
    
    # Road Type 1: 'Block'
    1: {'max_accel': 7.0, 'min_accel': -7.0, 'max_turn': 29.0},
    
    # Road Type 2: 'Concrete'
    2: {'max_accel': 8.5, 'min_accel': -8.5, 'max_turn': 27.5},
    
    # Road Type 3: 'Kanker' (Most tolerant to bumps and turns)
    3: {'max_accel': 6.0, 'min_accel': -6.0, 'max_turn': 32.0} # Increased turn threshold dramatically
}

ROAD_TYPE_MAP = {0: 'Bitumin', 1: 'Block', 2: 'Concrete', 3: 'Kanker'}

def run_rash_driving_detection(verbose=True):
    """
    Loads the model and simulates rash driving detection with calibrated rules.
    """
    print("Loading trained model...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading data for simulation...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at '{DATA_PATH}'.")
        return

    feature_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    unscaled_df = df.copy()

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    
    print("\n--- Starting Driving Simulation ---\n")

    for i in range(0, len(df) - CHUNK_SIZE, CHUNK_SIZE):
        scaled_chunk = df.iloc[i : i + CHUNK_SIZE]
        unscaled_chunk = unscaled_df.iloc[i : i + CHUNK_SIZE]
        
        model_input = scaled_chunk[feature_columns].values.reshape(1, CHUNK_SIZE, len(feature_columns))
        
        prediction = model.predict(model_input, verbose=0)
        predicted_road_id = np.argmax(prediction)
        
        rules = RASH_DRIVING_RULES.get(predicted_road_id)
        road_name = ROAD_TYPE_MAP.get(predicted_road_id, "Unknown")
        
        print(f"Time Segment {i//CHUNK_SIZE + 1}: Predicted Road Type -> {road_name}")
        
        rash_events_found = []
        for index, row in unscaled_chunk.iterrows():
            accel_x = row['acc_x']
            gyro_z = row['gyro_z']

            if accel_x > rules['max_accel']:
                event = f"Harsh Acceleration (value: {accel_x:.2f} > {rules['max_accel']})"
                rash_events_found.append(event)
            if accel_x < rules['min_accel']:
                event = f"Harsh Braking (value: {accel_x:.2f} < {rules['min_accel']})"
                rash_events_found.append(event)
            if abs(gyro_z) > rules['max_turn']:
                event = f"Sharp Turn (value: {abs(gyro_z):.2f} > {rules['max_turn']})"
                rash_events_found.append(event)

        if rash_events_found:
            print(f"  -> \033[91mALERT: Rash Driving Detected!\033[0m")
            if verbose:
                for event in rash_events_found:
                    print(f"     \033[93m- {event}\033[0m")
        else:
            print("  -> Driving: Normal")
        print("-" * 60)

if __name__ == '__main__':
    run_rash_driving_detection()
