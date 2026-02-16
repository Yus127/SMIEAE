import os
import pandas as pd
from pathlib import Path

def categorize_sensor(filename):
    """
    Categorize sensor based on filename and return target name. FROM ESFM
    
    Args:
        filename: Name of the CSV file
    
    Returns:
        Target category name
    """
    sensor_mapping = {
        'Nodo_Oficina': 'Nodo',
        'Nodo_3SP': 'Nodo',
        'Puerta_3SP': 'Prueba 3SP',
        'Lab_3SP': 'Sensor Lab 3SP',
        'Puerta_C006': 'Radar',
        'Sensirion_C006': 'Ambiental',
        'Puerta_106': 'ESFM_Puerta',
        'Contrario_106': 'ESFM_Contrario'
    }
    
    for key, value in sensor_mapping.items():
        if key in filename:
            return value
    
    return 'Unknown'

def read_sensor_data(base_path):
    """
    Read all CSV files from the directory structure and combine them by sensor type.
    
    Args:
        base_path: Root directory containing the month folders
    
    Returns:
        Dictionary with DataFrames for each sensor type
    """
    
    # Initialize dictionary to store dataframes by sensor type
    sensor_data = {}
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Determine sensor category
                    sensor_category = categorize_sensor(file)
                    
                    # Add metadata
                    df['source_file'] = file
                    df['sensor_category'] = sensor_category
                    
                    # Extract date from filename if possible
                    try:
                        date_part = file.split('_')[-1].replace('.csv', '')
                        df['date'] = date_part
                    except:
                        df['date'] = 'unknown'
                    
                    # Add to corresponding category
                    if sensor_category not in sensor_data:
                        sensor_data[sensor_category] = []
                    
                    sensor_data[sensor_category].append(df)
                    
                    print(f" Loaded: {file} → {sensor_category}")
                    
                except Exception as e:
                    print(f" Error reading {file}: {e}")
    
    # Combine dataframes for each sensor type
    combined_data = {}
    for sensor_type, dfs in sensor_data.items():
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Sort by timestamp if the column exists
            if 'timestamp' in combined_df.columns:
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            combined_data[sensor_type] = combined_df
    
    return combined_data

def main():
    base_path = '/Users/YusMolina/Downloads/smieae/data/original_data/ESFM'
    output_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/ESFM'
    
    print("Reading sensor data...")
    
    # Read all sensor data
    data = read_sensor_data(base_path)
    
    print(f"\nSummary:")
    print("-" * 60)
    
    total_records = 0
    for sensor_type, df in data.items():
        record_count = len(df)
        total_records += record_count
        print(f"{sensor_type}: {record_count} records")
    
    print(f"\nTotal records: {total_records}")
    print(f"Total sensor types: {len(data)}")
    
    # Display sample data for each sensor type
    print("\n" + "=" * 60)
    print("Sample Data:")
    
    for sensor_type, df in data.items():
        print(f"\n--- {sensor_type} ---")
        print(df.head(3))
    
    print("\n" + "=" * 60)
    save_combined = input("\nSave combined data to CSV files? (y/n): ")
    
    if save_combined.lower() == 'y':
        os.makedirs(output_path, exist_ok=True)
        
        for sensor_type, df in data.items():
            # Clean sensor type name for filename
            filename = sensor_type.replace(' ', '_').replace('/', '_')
            output_file = os.path.join(output_path, f'combined_{filename}.csv')
            
            df.to_csv(output_file, index=False)
            print(f" Saved: {output_file}")
        
        print(f"\nAll files saved to: {output_path}")
    
    return data

if __name__ == "__main__":
    sensor_data = main()



