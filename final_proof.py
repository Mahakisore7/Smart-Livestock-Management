import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
goat_file = "D:/Academic/SEM 4/Communication and IoT/Datasets_Motion/G1.csv"
cow_file  = "D:/Academic/SEM 4/Communication and IoT/Datasets_Motion/archive/data.csv"

def process_animal(file_path, animal_type):
    try:
        df = pd.read_csv(file_path)
        
        # --- 1. HANDLE GOAT DATA (200 Hz, Columns: ax, ay, az) ---
        if animal_type == "GOAT":
            # Calculate Energy: sqrt(ax^2 + ay^2 + az^2)
            # The PDF says columns are 'ax', 'ay', 'az'
            df['Energy'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
            
            # This data is 200Hz (too fast). We take every 20th row 
            # to bring it down to 10Hz (approx) so it matches the Cow.
            df = df.iloc[::20, :].reset_index(drop=True)
            
        # --- 2. HANDLE COW DATA (10 Hz, Columns: acc_x, acc_y, acc_z) ---
        elif animal_type == "COW":
            # The Image says columns are 'acc_x', 'acc_y', 'acc_z'
            df['Energy'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

        # Take first 300 points (which is now ~30 seconds for both)
        energy_data = df['Energy'].head(300)
        
        # Calculate Baseline (First 100 points as "Calibration")
        baseline = energy_data.head(100).mean()
        
        return energy_data, baseline

    except Exception as e:
        print(f"Error processing {animal_type}: {e}")
        return None, None

# --- EXECUTE ---
print("Reading Data...")
goat_energy, goat_base = process_animal(goat_file, "GOAT")
cow_energy, cow_base   = process_animal(cow_file, "COW")

# --- PLOT ---
if goat_energy is not None and cow_energy is not None:
    plt.figure(figsize=(10, 6))
    
    # Plot Goat (High Energy)
    plt.plot(goat_energy, color='#ff7f0e', alpha=0.8, label='Goat Activity (High)')
    plt.axhline(y=goat_base, color='#d62728', linestyle='--', linewidth=2, label=f'Goat Threshold ({goat_base:.2f})')
    
    # Plot Cow (Low Energy)
    plt.plot(cow_energy, color='#1f77b4', alpha=0.8, label='Cow Activity (Low)')
    plt.axhline(y=cow_base, color='#17becf', linestyle='--', linewidth=2, label=f'Cow Threshold ({cow_base:.2f})')
    
    plt.title("PROOF: Adaptive Baselines for Different Species")
    plt.xlabel("Time (Samples @ 10Hz)")
    plt.ylabel("Motion Intensity (g-force)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    print("Graph generated! Save this image.")
    plt.show()