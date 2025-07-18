import os
import glob
import pandas as pd

# --- Configuration ---
RAW_DATA_PATH = 'Rash_Driving_Project/0_Data/raw/'
PROCESSED_DATA_PATH = 'Rash_Driving_Project/0_Data/processed/'
OUTPUT_FILENAME = os.path.join(PROCESSED_DATA_PATH, 'balanced_road_data.csv')

def create_processed_dataset():
    """
    Loads all raw CSVs, skips the header, selects the correct sensor data columns
    by position, combines them, and saves a single clean file.
    """
    all_files = glob.glob(os.path.join(RAW_DATA_PATH, "*/*.csv"))
    
    if not all_files:
        print(f"Error: No CSV files found in '{RAW_DATA_PATH}'. Please check the path.")
        return

    df_list = []
    road_type_map = {'Bitumin': 0, 'Block': 1, 'Concrete': 2, 'Kanker': 3}
    
    # Define the standard column names we will assign.
    standard_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    print(f"Found {len(all_files)} files to process...")

    for file_path in all_files:
        try:
            # Read the CSV, skipping the first row (header).
            df = pd.read_csv(file_path, header=None, skiprows=1)
            
            if df.empty:
                continue

            # --- THE KEY CHANGE IS HERE ---
            # Select columns by their position:
            # We skip the first 3 columns (ID, SrNo, Timestamp) and take the next 6.
            df = df.iloc[:, 3:9]
            
            # Assign our standard names to these 6 columns.
            df.columns = standard_columns
            
            # Get the road type label from the folder name.
            road_type_name = os.path.basename(os.path.dirname(file_path))
            if road_type_name in road_type_map:
                df['label'] = road_type_map[road_type_name]
                df_list.append(df)
                
        except Exception as e:
            print(f"Could not process file {file_path}: {e}")
            
    if not df_list:
        print("\nError: No data was successfully processed.")
        return

    # Combine all the processed dataframes.
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"\nSuccessfully combined data from {len(df_list)} files.")
    print(f"Total rows in combined data: {full_df.shape[0]}")
    
    # Convert all feature columns to numeric, coercing errors to NaN.
    for col in standard_columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # Drop any rows that might have missing or non-numeric values.
    full_df.dropna(inplace=True)
    print(f"Total rows after cleaning: {full_df.shape[0]}")
    
    # Save the final, standardized dataset.
    print(f"Saving processed data to '{OUTPUT_FILENAME}'...")
    full_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print("\n✅ Preprocessing complete! The dataset is now ready.")

if __name__ == '__main__':
    create_processed_dataset()
