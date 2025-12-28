import pandas as pd
import os

# Define paths
cow_path = "H:\\Datasets_Motion\\1_Walking_1319_20240513_120009.csv"
goat_path = "H:\Datasets_Motion\G1.csv" 

def inspect_file(file_path, animal):
    print(f"--- CHECKING {animal} DATA ---")
    if os.path.exists(file_path):
        try:
            # Read just the first 3 rows to see headers
            df = pd.read_csv(file_path, nrows=3)
            print(f"Found file! Here are the column names:")
            print(list(df.columns))
            print("\nFirst row of data:")
            print(df.iloc[0].values)
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"WAITING... File not found: {file_path}")
    print("\n" + "="*30 + "\n")

# Run inspection
inspect_file(cow_path, "COW")
inspect_file(goat_path, "GOAT")