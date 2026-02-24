# Livestock AI Sentinel: Agentic Version

An autonomous, multi-cow health monitoring system that leverages Sensor Fusion (Audio, Motion, Temperature) and Generative AI to provide real-time clinical diagnostics and emergency protocols for farmers.

## 📂 Project Structure

All files are located in the ~/livestock_ai/livestock_agent directory.

```
livestock_ai/
├── livestock_env/             # Python Virtual Environment (outside code folder)
└── livestock_agent/           # Main Agentic Package
    ├── triple_fusion_agent3.py # Backend: Sensing & AI Reasoning
    ├── dashboard.py           # Frontend: Streamlit Dashboard
    ├── setup_db.py            # Script to initialize fresh Database
    ├── livestock_agent.db     # SQLite Database (Auto-generated)
    ├── cow_dataset.csv        # Textbook: Disease Pattern Reference
    ├── protocols.json         # Knowledge Base: Emergency Procedures
    ├── 1.tflite               # Edge AI Model: Cough Detection
    ├── yamnet_class_map.csv   # Label Map for Audio AI
    └── requirements.txt       # List of Python dependencies
```

## 🚀 Execution Instructions

### 1. Access and Environment Setup

Navigate to the root directory and activate your specific virtual environment:

```bash
cd ~/livestock_ai
source livestock_env/bin/activate
cd livestock_agent
```

### 2. Database Initialization

If you are running this for the first time or want to clear old data, run the setup script:

```bash
python setup_db.py
```

This ensures the livestock_agent.db has the correct columns for multi-cow scalability.

### 3. Running the Backend (The "Sense-Think" Loop)

Open a terminal window and start the sensor monitoring. This script will prompt you for a Cow ID to enable herd scalability.

```bash
python triple_fusion_agent3.py
```

**Prompt:** Enter Cow ID (e.g., Cow_01)

**Action:** Type your ID and press Enter. The agent is now monitoring and matching patterns against the cow_dataset.csv.

### 4. Running the Frontend (The "Clinical Command Center")

Open a new terminal tab, activate the environment again as shown in Step 1, and run:

```bash
streamlit run dashboard.py --server.headless true
```

**Access:** Open the URL provided (usually http://<Your-Pi-IP>:8501) on your laptop browser.

**Interaction:** Use the Sidebar to switch between different Cow IDs to see their specific health trends and AI alerts.

## 🧠 System Logic

- **Audio AI:** Uses the 1.tflite model to "hear" coughs in real-time via the microphone.
- **Pattern Matcher:** Every 60 seconds, the backend fuses Temp/Cough/Activity data and finds the closest match in cow_dataset.csv.
- **Agentic Reasoning:** If an anomaly is found, the Gemini AI Agent retrieves the specific protocol from protocols.json and writes a 3-sentence conversational alert for the user.

## ⚠️ Troubleshooting

- **"Stopping..." Error:** Ensure your microphone is connected and no other script is using the I2C bus (MPU6050).
- **"Waiting for data..." on Dashboard:** The backend takes 60 seconds to save the first data point. Wait for the terminal to print ✅ Data saved to Database.
