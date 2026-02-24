import os, smbus, time, math, threading, sqlite3, json, ctypes
import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert
import pandas as pd
from gpiozero import DigitalInputDevice, DigitalOutputDevice
from google import genai

# --- 1. CONFIG & SYSTEM SILENCING ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TFLITE_XNNPACK_DELEGATE'] = '0'

# Ask for Cow ID for Scalability
COW_ID = input("Enter Cow ID (e.g., Cow_01): ").strip() or "Unknown_Cow"

GEMINI_API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=GEMINI_API_KEY)

# Detect Model Name
try:
    available_models = [m.name for m in client.models.list()]
    ACTIVE_MODEL = "gemini-1.5-flash" if "gemini-1.5-flash" in available_models else available_models[0]
except:
    ACTIVE_MODEL = "gemini-1.5-flash"

DB_NAME = 'livestock_agent.db'
MODEL_PATH = "1.tflite"
LABELS_PATH = "yamnet_class_map.csv"
WINDOW_SIZE = 15600
MPU_ADDR = 0x68
NTC_PIN = 18

temp_buffer, activity_buffer = [], []
cough_events = 0
status = {"live_temp": 38.5, "cough_latched": False, "act": 0.0, "baseline": 0.000022}

# --- 2. CSV PATTERN MATCHER ---
def match_disease_from_csv(temp, cough_count, activity_level):
    try:
        df = pd.read_csv('cow_dataset.csv')
        df['temp_val'] = df['Body_Temperature'].str.extract('(\d+\.\d+)').astype(float)
        has_cough = "Yes" if cough_count > 0 else "No"
        is_lethargic = activity_level == "Low (Lethargic)"
        
        filtered_df = df[df['Coughing'] == has_cough].copy()
        def score(row):
            s = [str(row[f'Symptom_{i}']) for i in range(1, 5)]
            val = 0
            if is_lethargic and any("Lethargy" in i for i in s): val += 3
            if temp > 39.2 and any("Fever" in i for i in s): val += 2
            return val

        filtered_df['score'] = filtered_df.apply(score, axis=1)
        filtered_df['t_diff'] = (filtered_df['temp_val'] - temp).abs()
        best = filtered_df.sort_values(by=['score', 't_diff'], ascending=[False, True]).iloc[0]
        return best['Disease_Prediction']
    except: return "Healthy" if temp < 39.0 else "General Fever"

# --- 3. AGENT REASONING (CONVERSATIONAL) ---
def ask_agent(temp, coughs, act, diagnosis):
    try:
        with open('protocols.json', 'r') as f: protocols = f.read()
        prompt = (f"As a Vet AI, alert a farmer: {COW_ID} has {temp}C, {coughs} coughs, {act}. "
                  f"Textbook Match: {diagnosis}. Protocols: {protocols}. "
                  "Write a professional, calm, 3-sentence alert. No bullets.")
        response = client.models.generate_content(model=ACTIVE_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        return f"System Alert: {COW_ID} shows signs of {diagnosis}. Please isolate and check temperature immediately."

# --- 4. HARDWARE & DB ---
def save_to_db(temp, coughs, act, diag, advice):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("INSERT INTO health_logs (cow_id, animal, avg_temp, cough_count, activity_level, diagnosis, agent_advice) VALUES (?,?,?,?,?,?,?)",
                 (COW_ID, "Cow", temp, coughs, act, diag, advice))
    conn.commit(); conn.close()

def get_ntc_temp():
    d = DigitalOutputDevice(NTC_PIN, active_high=False); d.on(); time.sleep(0.01); d.close()
    c = DigitalInputDevice(NTC_PIN, pull_up=False); start = time.perf_counter()
    while not c.is_active and time.perf_counter()-start < 0.1: pass
    dur = time.perf_counter() - start; c.close()
    ratio = max(dur / status["baseline"], 0.001)
    calc = 38.0 + (-math.log(ratio) * 18.5)
    return round(max(min(calc, 42.5), 36.5), 1)

def sensor_loop():
    bus = smbus.SMBus(1); bus.write_byte_data(MPU_ADDR, 0x6b, 0)
    resting_g = sum([math.sqrt((((bus.read_byte_data(MPU_ADDR, 0x3b)<<8)|bus.read_byte_data(MPU_ADDR, 0x3c))-65536 if ((bus.read_byte_data(MPU_ADDR, 0x3b)<<8)|bus.read_byte_data(MPU_ADDR, 0x3c))>32768 else ((bus.read_byte_data(MPU_ADDR, 0x3b)<<8)|bus.read_byte_data(MPU_ADDR, 0x3c)))/16384.0)**2 for _ in range(10)])/10 # Simple magnitude
    while True:
        status["live_temp"] = get_ntc_temp(); time.sleep(0.5)

def audio_loop():
    interpreter = litert.Interpreter(model_path=MODEL_PATH); interpreter.allocate_tensors()
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    while True:
        data = np.frombuffer(stream.read(1024, exception_on_overflow=False), dtype=np.int16).astype(np.float32)/32768.0
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.zeros(15600, np.float32)) # Simple trigger
        interpreter.invoke()
        if interpreter.get_tensor(interpreter.get_output_details()[0]['index']).flatten()[0] > 0.35: status["cough_latched"] = True

# --- 5. MAIN ---
threading.Thread(target=sensor_loop, daemon=True).start()
threading.Thread(target=audio_loop, daemon=True).start()

print(f"🚀 MONITORING {COW_ID}...")
try:
    while True:
        temp_buffer.append(status["live_temp"])
        if status["cough_latched"]: cough_events += 1; status["cough_latched"] = False
        if len(temp_buffer) >= 6:
            avg_t = round(sum(temp_buffer)/6, 1)
            avg_a = "Low (Lethargic)" if status["act"] < 0.08 else "Normal"
            diag = match_disease_from_csv(avg_t, cough_events, avg_a)
            advice = ask_agent(avg_t, cough_events, avg_a, diag) if (diag != "Healthy" or avg_t > 39.5) else "Cow is healthy."
            save_to_db(avg_t, cough_events, avg_a, diag, advice)
            print(f"📊 {COW_ID} Summary: {avg_t}C | Diagnosis: {diag}")
            temp_buffer, cough_events = [], 0
        time.sleep(10)
except: print("Stopping...")
