import os
import pandas as pd
from pathlib import Path

def read_sensor_data(base_path):
    """
    Read all CSV files from the directory structure and combine them.
    
    Args:
        base_path: Root directory containing the numbered folders (05, 09, 10, 11, etc.)
    
    Returns:
        Dictionary with two DataFrames: 'puerta' and 'contrario'
    """
    
    # Initialize lists to store dataframes
    puerta_dfs = []
    contrario_dfs = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    # Add source information
                    df['source_file'] = file
                    df['date'] = file.split('_')[-1].replace('.csv', '')
                    
                    # Categorize by sensor type
                    if 'Puerta' in file:
                        df['sensor_type'] = 'puerta'
                        puerta_dfs.append(df)
                    elif 'Contrario' in file:
                        df['sensor_type'] = 'contrario'
                        contrario_dfs.append(df)
                    
                    print(f"✓ Loaded: {file}")
                    
                except Exception as e:
                    print(f"✗ Error reading {file}: {e}")
    
    # Combine all dataframes
    puerta_combined = pd.concat(puerta_dfs, ignore_index=True) if puerta_dfs else pd.DataFrame()
    contrario_combined = pd.concat(contrario_dfs, ignore_index=True) if contrario_dfs else pd.DataFrame()
    
    # Sort by timestamp if the column exists
    if not puerta_combined.empty and 'timestamp' in puerta_combined.columns:
        puerta_combined = puerta_combined.sort_values('timestamp').reset_index(drop=True)
    
    if not contrario_combined.empty and 'timestamp' in contrario_combined.columns:
        contrario_combined = contrario_combined.sort_values('timestamp').reset_index(drop=True)
    
    return {
        'puerta': puerta_combined,
        'contrario': contrario_combined
    }

def main():
    # Set your base directory path here
    base_path = '.'  # Current directory, change as needed
    
    print("Reading sensor data...")
    print("-" * 50)
    
    # Read all sensor data
    data = read_sensor_data(base_path)
    
    print("-" * 50)
    print(f"\nSummary:")
    print(f"Puerta sensor records: {len(data['puerta'])}")
    print(f"Contrario sensor records: {len(data['contrario'])}")
    
    # Display sample data
    if not data['puerta'].empty:
        print("\n--- Puerta Sensor Sample ---")
        print(data['puerta'].head())
    
    if not data['contrario'].empty:
        print("\n--- Contrario Sensor Sample ---")
        print(data['contrario'].head())
    
    # Optional: Save combined data to new CSV files
    save_combined = input("\nSave combined data to CSV files? (y/n): ")
    if save_combined.lower() == 'y':
        if not data['puerta'].empty:
            data['puerta'].to_csv('combined_puerta_sensor.csv', index=False)
            print("✓ Saved: combined_puerta_sensor.csv")
        
        if not data['contrario'].empty:
            data['contrario'].to_csv('combined_contrario_sensor.csv', index=False)
            print("✓ Saved: combined_contrario_sensor.csv")
    
    return data

if __name__ == "__main__":
    sensor_data = main()
