import os, smbus, time, math, threading, sqlite3, json, joblib
import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert
import pandas as pd
from gpiozero import DigitalOutputDevice, DigitalInputDevice
from google import genai

# --- 1. SYSTEM CONFIGURATION ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 🔥 API SETUP
GEMINI_API_KEY = "AIzaSyCv_h9xvjhmyoLvknaVjK20NbxBcc6WZHA"
client = genai.Client(api_key=GEMINI_API_KEY)

# Audio Parameters (Fixed at 16kHz for USB PnP Hardware Compatibility)
CAPTURE_RATE = 16000 
MODEL_INPUT_SIZE = 15600
FRAMES_PER_BUFFER = 1024
CONFIDENCE_THRESHOLD = 0.30

# Hardware Pins & Model Paths
NTC_PIN = 18
MPU_ADDR = 0x68
MODEL_PATH = "1.tflite"
LABELS_PATH = "yamnet_class_map.csv"
CLINICAL_MODEL = "clinical_model.pkl"
CLASS_MAP = "disease_classes.pkl"

# --- 2. GLOBAL TRACKERS ---
# Baseline 0.000115 tuned for ~37.5°C room temp
status = {"temp": 37.5, "act": 0.0, "coughs": 0, "baseline": 0.000115, "g_offset": 1.0}
COW_ID = "Unknown_Cow" 
lock = threading.Lock()
p = pyaudio.PyAudio()

# --- 3. QUOTA-PROOF REASONING ENGINE ---
def ask_agent(temp, coughs, act_str, top_results, cow_id):
    try:
        with open('protocols.json', 'r') as f:
            protocols = json.load(f)

        primary = top_results[0][0]
        p_data = protocols.get(primary)
        
        if not p_data:
            for key in protocols.keys():
                if primary.lower() in key.lower() or key.lower() in primary.lower():
                    p_data = protocols[key]
                    break

        if not p_data:
            p_data = protocols.get("Healthy", {"exams": ["General check"], "actions": ["Monitor"], "priority": "LOW"})

        try:
            prompt = (
                f"Senior Vet AI for {cow_id}. Vitals: {temp}°C, {coughs} coughs, Activity: {act_str}.\n"
                f"Prediction: {primary}.\n"
                f"Protocol: {json.dumps(p_data)}.\n\n"
                "Explain the suspicion, provide the physical exam checklist, and state isolation steps."
            )
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return response.text.strip()
        except Exception:
            exams = "\n- ".join(p_data.get('exams', []))
            return f"⚠️ [OFFLINE] Suspect: {primary} for {cow_id}. Checklist: {exams}"
    except Exception as e:
        return f"CRITICAL: {top_results[0][0]} detected. (Error: {str(e)})"

# --- 4. SENSOR LOOP (TEMP & MOTION CALIBRATED) ---
def sensor_loop():
    global status
    bus = smbus.SMBus(1)
    try: bus.write_byte_data(MPU_ADDR, 0x6b, 0)
    except: pass

    # Startup Calibration
    cal_samples = []
    print("⚖️ Calibrating Motion sensor (Do not move)...")
    for _ in range(10):
        def r_cal(reg):
            v = (bus.read_byte_data(MPU_ADDR, reg) << 8) | bus.read_byte_data(MPU_ADDR, reg+1)
            return v - 65536 if v > 32768 else v
        mx, my, mz = r_cal(0x3b)/16384.0, r_cal(0x3d)/16384.0, r_cal(0x3f)/16384.0
        cal_samples.append(math.sqrt(mx**2 + my**2 + mz**2))
        time.sleep(0.5)

    status["g_offset"] = sum(cal_samples) / len(cal_samples)
    print(f"✅ Motion Calibrated. Offset: {status['g_offset']:.4f}")

    while True:
        try:
            d = DigitalOutputDevice(NTC_PIN, active_high=False); d.on(); time.sleep(0.01); d.close()
            c = DigitalInputDevice(NTC_PIN, pull_up=False); s = time.perf_counter()
            while not c.is_active and time.perf_counter()-s < 0.1: pass
            dur = time.perf_counter() - s; c.close()

            ratio = max(dur / status["baseline"], 0.01)
            raw_t = 37.4 + (-math.log(ratio) * 3.5)

            status["temp"] = (status["temp"] * 0.8) + (raw_t * 0.2)
            if status["temp"] > 41.5: status["temp"] = 41.2
            if status["temp"] < 36.5: status["temp"] = 37.2
            status["temp"] = round(status["temp"], 1)

            def r(reg):
                v = (bus.read_byte_data(MPU_ADDR, reg) << 8) | bus.read_byte_data(MPU_ADDR, reg+1)
                return v - 65536 if v > 32768 else v
            mag = math.sqrt((r(0x3b)/16384.0)**2 + (r(0x3d)/16384.0)**2 + (r(0x3f)/16384.0)**2)
            linear_acc = abs(mag - status["g_offset"])
            status["act"] = 0.0 if linear_acc < 0.02 else linear_acc
        except: pass
        time.sleep(0.5)

# --- 5. AUDIO LOOP (USB PnP FIX) ---
def audio_loop():
    global status
    try:
        interpreter = litert.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        i_idx = interpreter.get_input_details()[0]['index']
        o_idx = interpreter.get_output_details()[0]['index']
    except Exception as e:
        print(f"❌ Audio Error: YAMNet Load Failed ({e})")
        return

    # Try common hardware rates until one works
    supported_rates = [44100, 48000, 16000]
    stream = None
    actual_rate = 0

    for rate in supported_rates:
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                input_device_index=2,
                frames_per_buffer=2048
            )
            actual_rate = rate
            print(f"✅ USB Mic connected successfully at {rate}Hz")
            break
        except:
            continue

    if not stream:
        print("❌ Audio Error: Could not find a supported sample rate for this USB Mic.")
        return

    # Calculate how many samples we need for 1 second of audio
    # YAMNet needs 16000 samples per second
    audio_buffer = np.zeros(actual_rate, dtype=np.float32)
    labels_df = pd.read_csv(LABELS_PATH)
    C_IDX = labels_df[labels_df['display_name'] == 'Cough'].index[0]
    last_detection_time = 0

    while True:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(samples))
            audio_buffer[-len(samples):] = samples

            # --- DOWNSAMPLING MATH ---
            # If rate is 48000, we take every 3rd sample to get 16000
            # If rate is 44100, we use a slice to fit the model size
            step = actual_rate // 16000
            if step > 1:
                input_data = audio_buffer[::step]
            else:
                input_data = audio_buffer

            # Final check to ensure the AI gets exactly 15600 samples
            input_data = np.ascontiguousarray(input_data[-MODEL_INPUT_SIZE:]).astype(np.float32)

            interpreter.set_tensor(i_idx, input_data)
            interpreter.invoke()

            scores = interpreter.get_tensor(o_idx).flatten()
            if scores[C_IDX] > CONFIDENCE_THRESHOLD:
                curr = time.time()
                if curr - last_detection_time > 1.5:
                    with lock: status["coughs"] += 1
                    last_detection_time = curr
                    print(f"\n🚨 {COW_ID}: COUGH DETECTED! (Score: {scores[C_IDX]:.2f})")
        except Exception:
            continue

# --- 6. MAIN ENGINE ---
if __name__ == "__main__":
    print("\n🐄 --- HERD HEALTH SENTINEL v13.4 ---")
    COW_ID = input("Enter Cow ID: ").strip() or "Cow_Default"
    
    clf = joblib.load(CLINICAL_MODEL)
    classes = joblib.load(CLASS_MAP)

    threading.Thread(target=sensor_loop, daemon=True).start()
    threading.Thread(target=audio_loop, daemon=True).start()

    print(f"🚀 MONITORING {COW_ID} | LIVE DIAGNOSTICS ACTIVE")
    t_buf, a_buf = [], []

    while True:
        time.sleep(3.33)
        t_buf.append(status["temp"]); a_buf.append(status["act"])
        print(f"🌡️ {status['temp']}°C | 🔊 C:{status['coughs']} | 🏃 A:{status['act']:.4f}", end='\r')

        if len(t_buf) >= 6:
            avg_t, avg_a = round(sum(t_buf)/len(t_buf), 1), sum(a_buf)/len(a_buf)
            l_bit = 1 if avg_a < 0.05 else 0
            with lock: c_count = status["coughs"]; status["coughs"] = 0

            in_df = pd.DataFrame([{'lethargy': l_bit, 'coughing': 1 if c_count > 0 else 0, 'temperature': avg_t}])
            probs = clf.predict_proba(in_df)[0]
            results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
            top_3 = [[d, round(p*100, 1)] for d, p in results if p > 0.05][:3]

            if top_3[0][0] == "Healthy":
                advice = f"{COW_ID} vitals normal."
            else:
                advice = ask_agent(avg_t, c_count, "Lethargic" if l_bit else "Normal", top_3, COW_ID)

            try:
                DB_PATH = "/home/d11/livestock_ai/livestock_agent/livestock_agent.db"
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO health_logs (cow_id, animal, avg_temp, cough_count, activity_level, diagnosis, agent_advice) VALUES (?, 'Cow', ?, ?, ?, ?, ?)", 
                               (COW_ID, avg_t, c_count, "Lethargic" if l_bit else "Normal", top_3[0][0], advice))
                conn.commit(); conn.close()
            except: pass

            print(f"\n📊 REPORT for {COW_ID}: {avg_t}°C | Coughs: {c_count} | Suspect: {top_3[0][0]}")
            print(f"🤖 AI: {advice}\n" + "-"*40)
            t_buf, a_buf = [], []
