# 🐄 Livestock Health Sentinel: Triple Fusion Core AI

An autonomous, edge-computing livestock health monitoring system powered
by Raspberry Pi 5, designed to deliver real-time clinical insights using
Triple Sensor Fusion.

This system acts as a Digital Veterinary Sentinel, cross-analyzing
acoustic, kinetic, and thermal data to generate intelligent disease
probability assessments directly at the edge.

------------------------------------------------------------------------

## 🚀 Triple Fusion Architecture

The system integrates three biological signal streams:

### 🎧 1. Acoustic Intelligence (Audio AI)

-   Model: YAMNet (Quantized LiteRT)
-   Detects respiratory distress patterns (coughs)
-   Input: Bluetooth microphone
-   Output: Cough probability scores

------------------------------------------------------------------------

### 📈 2. Kinetic Analysis (Motion Intelligence)

-   Sensor: MPU6050 (I2C Accelerometer)

-   Computation:

    Vector Magnitude = √(X² + Y² + Z²)

-   Detects:

    -   Lethargy
    -   Abnormal tremors
    -   Reduced locomotion

------------------------------------------------------------------------

### 🌡️ 3. Thermal Profiling (Heat Intelligence)

-   Sensor: 10kΩ NTC Thermistor
-   Circuit: RC timing-based analog measurement
-   Detects:
    -   Fever
    -   Hypothermia
    -   Temperature deviations from baseline

------------------------------------------------------------------------

# 🧠 Autonomous Reasoning Engine

Core File: triple_fusion1.py

Every 10 seconds, the system generates a Clinical Snapshot:

1.  Aggregates sensor data
2.  Performs fuzzy matching against:
    -   cow_dataset.csv
    -   goat_dataset.csv
3.  Outputs:
    -   Probability match for diseases
    -   Clinical examination suggestions

------------------------------------------------------------------------

## 🧬 Example Output

--- Clinical Snapshot --- Cough Score: 0.82 Motion Index: Low
Temperature: 40.1°C

Probable Condition: Pneumonia (78% match) Suggested Exam: - Check nasal
discharge - Monitor respiration rate - Inspect lung sounds

------------------------------------------------------------------------

# 🛠️ Hardware Requirements

-   Raspberry Pi 5 (8GB recommended)
-   MPU6050 (Accelerometer + Gyroscope)
-   10k NTC Thermistor + 0.1µF Capacitor
-   Bluetooth Microphone

------------------------------------------------------------------------

# 📦 Installation

## 1️⃣ Clone Repository

git clone https://github.com/your-repo/livestock-health-sentinel.git cd
livestock-health-sentinel

## 2️⃣ Install Dependencies

pip install -r requirements.txt

## 3️⃣ Run the Sentinel

python triple_fusion1.py

------------------------------------------------------------------------

# 🔮 Roadmap: Agentic AI Phase

### 🧠 AI Vet Agent

-   Context-aware treatment suggestions
-   Breed-specific disease modeling
-   Escalation alerts

### ⏳ Temporal Expansion

-   Move from 10-second snapshots
-   To 10-minute biological trend windows

### 🌐 Web Dashboard

-   Built with Streamlit
-   Real-time monitoring
-   Historical data visualization

------------------------------------------------------------------------

# 🎯 Mission

Enable early disease detection, reduce livestock mortality, and empower
farmers with AI-driven veterinary intelligence at the edge.
