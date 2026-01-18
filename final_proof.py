import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# --- CONFIGURATION ---
# ==========================================
cow_file_path = r"C:\Users\yashw_d6scpoj\Desktop\Academics\S4\IoT\Projects\Livestock\Codes\cow_1_data.csv"
goat_file_path = r"C:\Users\yashw_d6scpoj\Desktop\Academics\S4\IoT\Projects\Livestock\Codes\balanced_goat_dataset.csv"

# CONTROL HOW MUCH DATA TO SEE HERE:
SAMPLES_TO_SHOW = 3500  # <--- Change this to 1000, 5000, etc.

cow_cols = ['acc_x', 'acc_y', 'acc_z']
goat_cols = ['ax', 'ay', 'az']

# ==========================================
# --- THE LOGIC (Same as before) ---
# ==========================================
def get_processed_data(file_path, cols, animal_name):
    try:
        print(f"Processing {animal_name}...")
        df = pd.read_csv(file_path)
        
        # 1. Calculate Raw Energy
        x = df[cols[0]].values
        y = df[cols[1]].values
        z = df[cols[2]].values
        raw_energy = np.sqrt(x**2 + y**2 + z**2)
        
        # 2. Baseline & Threshold
        baseline = np.mean(raw_energy[:300])
        raw_threshold = baseline * 0.6 
        
        # 3. Normalize
        norm_energy = raw_energy - baseline
        norm_threshold = raw_threshold - baseline
        
        return norm_energy, norm_threshold
        
    except Exception as e:
        print(f"Error with {animal_name}: {e}")
        return None, None

# --- EXECUTE ---
cow_energy, cow_alert_line = get_processed_data(cow_file_path, cow_cols, "COW")
goat_energy, goat_alert_line = get_processed_data(goat_file_path, goat_cols, "GOAT")

# --- PLOT (SLICED) ---
if cow_energy is not None and goat_energy is not None:
    plt.figure(figsize=(14, 7))
    
    # === SLICING THE DATA ===
    # We create a temporary "view" of just the first N samples
    cow_view = cow_energy[:SAMPLES_TO_SHOW]
    goat_view = goat_energy[:SAMPLES_TO_SHOW]
    
    # --- PLOT COW ---
    plt.plot(cow_view, label='Cow Activity', color='blue', alpha=0.6, linewidth=1)
    plt.axhline(cow_alert_line, color='cyan', linestyle='--', linewidth=2, label='Cow Alert Threshold')
    
    # Cow Alerts (Calculated only on the viewed data)
    under_cow = cow_view < cow_alert_line
    # np.where returns indices, we plot dots at those indices
    alert_indices_cow = np.where(under_cow)[0]
    plt.scatter(alert_indices_cow, cow_view[alert_indices_cow], color='red', s=20, label='Cow Alert', zorder=5)

    # --- PLOT GOAT ---
    plt.plot(goat_view, label='Goat Activity', color='orange', alpha=0.6, linewidth=1)
    plt.axhline(goat_alert_line, color='darkred', linestyle='--', linewidth=2, label='Goat Alert Threshold')
    
    # Goat Alerts
    under_goat = goat_view < goat_alert_line
    alert_indices_goat = np.where(under_goat)[0]
    plt.scatter(alert_indices_goat, goat_view[alert_indices_goat], color='darkred', s=20, marker='x', label='Goat Alert', zorder=5)
    
    # --- DECORATION ---
    plt.axhline(0, color='black', linewidth=1, label='Baseline (0)')
    plt.title(f"Activity Monitor: First {SAMPLES_TO_SHOW} Samples", fontsize=16)
    plt.ylabel("Relative Intensity")
    plt.xlabel("Time (Samples)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.show()