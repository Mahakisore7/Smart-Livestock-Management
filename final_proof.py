import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CONFIGURATION (MATCHED TO YOUR DATA) ---
# ==========================================

# 1. FILE PATHS
cow_file_path = "H:\\Datasets_Motion\\1_Walking_1319_20240513_120009.csv"
goat_file_path = "H:\Datasets_Motion\G1.csv" 

# 2. COLUMN NAMES (From your Inspector Output)
# We use the MPU9250 columns for the cow
cow_cols = ['MPU9250_AX', 'MPU9250_AY', 'MPU9250_AZ']

# We use ax, ay, az for the goat
goat_cols = ['ax', 'ay', 'az']

# ==========================================
# --- THE LOGIC (Simulation of ESP32) ---
# ==========================================

def get_energy(file_path, cols, animal_name):
    try:
        print(f"Reading {animal_name} data...")
        df = pd.read_csv(file_path)
        
        # 1. Calculate Vector Magnitude (Total Energy)
        # Formula: sqrt(x^2 + y^2 + z^2)
        # We use .values ensures we handle numpy arrays correctly
        x = df[cols[0]].values
        y = df[cols[1]].values
        z = df[cols[2]].values
        
        energy = np.sqrt(x**2 + y**2 + z**2)
        
        # 2. Simulate "Baseline Learning" (First 300 data points)
        # In real life, this would be 3 days. Here, it's the first few seconds.
        learning_phase = energy[:300]
        baseline = np.mean(learning_phase)
        
        # 3. Simulate "Alert Threshold"
        # If activity drops below 60% of THIS animal's normal, we alert.
        alert_threshold = baseline * 0.6
        
        print(f"   -> {animal_name} Baseline Set: {baseline:.2f}")
        print(f"   -> {animal_name} Alert Threshold: {alert_threshold:.2f}")
        
        # Return only first 1000 points so the graph isn't messy
        return energy[:1000], baseline, alert_threshold
        
    except Exception as e:
        print(f"ERROR with {animal_name}: {e}")
        return None, None, None

# --- EXECUTE ---
cow_energy, cow_base, cow_thresh = get_energy(cow_file_path, cow_cols, "COW")
goat_energy, goat_base, goat_thresh = get_energy(goat_file_path, goat_cols, "GOAT")

# --- PLOT THE EVIDENCE ---
if cow_energy is not None and goat_energy is not None:
    plt.figure(figsize=(12, 6))
    
    # Plot Goat (High Energy)
    plt.plot(goat_energy, color='orange', alpha=0.6, label='Goat Motion (Raw)')
    plt.axhline(y=goat_base, color='red', linestyle='-', linewidth=2, label=f'Goat Normal ({goat_base:.1f})')
    plt.axhline(y=goat_thresh, color='red', linestyle=':', linewidth=2, label='Goat Alert Limit')
    
    # Plot Cow (Low Energy)
    plt.plot(cow_energy, color='blue', alpha=0.6, label='Cow Motion (Raw)')
    plt.axhline(y=cow_base, color='cyan', linestyle='-', linewidth=2, label=f'Cow Normal ({cow_base:.1f})')
    plt.axhline(y=cow_thresh, color='cyan', linestyle=':', linewidth=2, label='Cow Alert Limit')
    
    plt.title("Proof: Algorithm Auto-Adapts to Species (Cow vs Goat)", fontsize=14)
    plt.ylabel("Activity Level (g-force)")
    plt.xlabel("Time (Samples)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    print("Graph generated!")
    plt.show()