import pandas as pd
import os

# ==========================================
# --- USER CONFIGURATION ---
# ==========================================

# Define your files with their specific LABEL names
# Format: ("Label Name", r"File Path")
files_to_process = [
    ("Walking",  r"H:\Datasets_Motion\db-cow-walking-main\db-cow-walking-main\Walking\1_Walking_1319_20240513_120009.csv"),
    ("Resting",  r"H:\Datasets_Motion\db-cow-walking-main\db-cow-walking-main\Resting\33_Resting_4821_20240514_133644.csv"),
    ("Standing", r"H:\Datasets_Motion\db-cow-walking-main\db-cow-walking-main\Miscellaneous behaviors\10_Standing_4821_20240514_102311.csv"),
    ("Grazing",  r"H:\Datasets_Motion\db-cow-walking-main\db-cow-walking-main\Grazing\6_Grazing_4821_20240513_142048.csv")
]

# Output location
output_file = r"H:\Datasets_Motion\cow1_complete_dataset.csv"

# ==========================================
# --- MERGE LOGIC WITH LABELING ---
# ==========================================

def merge_and_label():
    print("--- STARTING MERGE & LABEL PROCESS ---\n")
    
    df_list = []
    total_rows = 0
    
    for label, file_path in files_to_process:
        print(f"Processing: {label}")
        print(f"   -> Path: {file_path}")
        
        if os.path.exists(file_path):
            try:
                # 1. Read the CSV
                df = pd.read_csv(file_path)
                
                # 2. ADD THE LABEL COLUMN (The Magic Step)
                df['label'] = label
                
                # 3. Add to list
                df_list.append(df)
                
                rows = len(df)
                total_rows += rows
                print(f"   -> Success! Loaded {rows} rows. Label assigned: '{label}'")
            except Exception as e:
                print(f"   -> ERROR: Could not read file. Reason: {e}")
        else:
            print(f"   -> WARNING: File not found!")
        print("-" * 30)

    # Combine everything
    if df_list:
        print("\nMerging all labeled files...")
        master_df = pd.concat(df_list, ignore_index=True)
        
        # Save to disk
        master_df.to_csv(output_file, index=False)
        
        print("\n" + "="*40)
        print("SUCCESS! MASTER FILE CREATED.")
        print(f"Saved to: {output_file}")
        print(f"Total Rows: {total_rows}")
        print(f"Columns: {list(master_df.columns)}")
        print("="*40)
    else:
        print("\nFAILED: No valid files were loaded.")

if __name__ == "__main__":
    merge_and_label()