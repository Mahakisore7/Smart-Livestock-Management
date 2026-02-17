import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert
import pandas as pd
import os

# --- 1. Configuration ---
MODEL_PATH = "1.tflite"
LABELS_PATH = "yamnet_class_map.csv"
WINDOW_SIZE = 15600 

# --- 2. Load Labels ---
labels_df = pd.read_csv(LABELS_PATH)
COUGH_INDEX = labels_df[labels_df['display_name'] == 'Cough'].index[0]

# --- 3. Setup Audio & Find Buds ---
p = pyaudio.PyAudio()
target_id = None

print("🔍 Scanning all audio devices...")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    name = info['name']
    print(f"ID {i}: {name}") # This helps us see what the Pi calls your buds
    if any(keyword in name for keyword in ["OnePlus", "bluez", "Buds", "Headset"]):
        target_id = i
        print(f"⭐ TARGET FOUND: Using ID {i} ({name})")
        break

if target_id is None:
    print("⚠️ No Buds found by name. Defaulting to ID 0.")
    target_id = 0

# --- 4. Load LiteRT Model ---
interpreter = litert.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open Bluetooth Stream
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, 
                input=True, input_device_index=target_id, frames_per_buffer=1024)

audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)

print("\n🚀 AI ACTIVE. Monitoring through Bluetooth...")

try:
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        audio_buffer = np.roll(audio_buffer, -len(samples))
        audio_buffer[-len(samples):] = samples
        
        # --- FIX: Match Model Dimensions ---
        # If model expects 1D [15600], we do not reshape to (1, -1)
        input_data = audio_buffer 
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        scores = interpreter.get_tensor(output_details[0]['index']).flatten()
        cough_score = scores[COUGH_INDEX]
        
        if cough_score > 0.3:
            print(f"🚨 COUGH! Confidence: {cough_score:.2%}")
        else:
            vol = np.max(np.abs(audio_buffer))
            print(f"🟢 Listening... Vol: {vol:.4f} Cough Score: {cough_score:.4f}", end='\r')

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    p.terminate()
