import smbus, time, math, threading, os, sys, ctypes, statistics
import numpy as np
import pyaudio
import ai_edge_litert.interpreter as litert
import pandas as pd
from gpiozero import DigitalInputDevice, DigitalOutputDevice
import telepot

# --- 1. TELEGRAM SETUP ---
BOT_TOKEN = '8294227754:AAFQPWg2OKFC2VCfywh_xDEomLnWyzZNYIQ'
CHAT_ID = '7376632488'
bot = telepot.Bot(BOT_TOKEN)

def send_to_telegram(text):
    """Isolated function to send text to phone without crashing AI."""
    def _async_send():
        try:
            bot.sendMessage(CHAT_ID, text)
        except:
            pass 
    threading.Thread(target=_async_send).start()

# --- 2. CONFIGURATION ---
MODEL_PATH = "1.tflite"
LABELS_PATH = "yamnet_class_map.csv"
WINDOW_SIZE = 15600
MPU_ADDR = 0x68
NTC_PIN = 18
LETHARGY_THRESHOLD = 0.08
SENSITIVITY = 8.5 

status = {"act": 0.0, "cough_latched": False, "live_temp": 38.0, "baseline": 0.000019}

# --- 3. HARDWARE INIT ---
bus = smbus.SMBus(1)
bus.write_byte_data(MPU_ADDR, 0x6b, 0)
labels_df = pd.read_csv(LABELS_PATH)
COUGH_INDEX = labels_df[labels_df['display_name'] == 'Cough'].index[0]

# --- PI 5 STABILITY FIX ---
interpreter = litert.Interpreter(model_path=MODEL_PATH, num_threads=1)
interpreter.allocate_tensors()
input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()

# --- 4. SENSOR LOGIC ---
def get_mag():
    def r(reg):
        h = bus.read_byte_data(MPU_ADDR, reg); l = bus.read_byte_data(MPU_ADDR, reg+1)
        v = (h << 8) | l
        return v - 65536 if v > 32768 else v
    ax, ay, az = r(0x3b)/16384.0, r(0x3d)/16384.0, r(0x3f)/16384.0
    return math.sqrt(ax**2 + ay**2 + az**2)

def get_ntc_duration():
    discharge = DigitalOutputDevice(NTC_PIN, active_high=False)
    discharge.on(); time.sleep(0.01); discharge.close()
    charge_check = DigitalInputDevice(NTC_PIN, pull_up=False)
    start = time.perf_counter()
    while not charge_check.is_active:
        if time.perf_counter() - start > 0.05: break
    duration = time.perf_counter() - start
    charge_check.close()
    return duration

def get_stable_temp(is_calib=False):
    samples = [get_ntc_duration() for _ in range(50)]
    samples.sort()
    avg_d = sum(samples[15:-15]) / len(samples[15:-15])
    if is_calib: return avg_d
    ratio = avg_d / status["baseline"]
    if ratio <= 0: ratio = 1.0
    temp_diff = -math.log(max(ratio, 0.0001)) * SENSITIVITY
    return round(38.0 + temp_diff, 1)

# --- 5. BACKGROUND THREADS ---
def sensor_loop():
    print("⚖️ Calibrating Motion and Thermistor...")
    resting_g = sum([get_mag() for _ in range(20)]) / 20.0
    status["baseline"] = sum([get_stable_temp(True) for _ in range(10)]) / 10.0
    while True:
        readings = [abs(get_mag() - resting_g) for _ in range(30)]
        status["act"] = sum([x if x > 0.05 else 0.0 for x in readings]) / len(readings)
        status["live_temp"] = get_stable_temp()
        time.sleep(0.5)

def audio_loop():
    p = pyaudio.PyAudio()
    target_id = 0
    for i in range(p.get_device_count()):
        dev_name = p.get_device_info_by_index(i)['name']
        if any(k in dev_name for k in ["OnePlus", "bluez", "Buds"]):
            target_id = i; break
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, input_device_index=target_id)
    audio_buffer = np.zeros(WINDOW_SIZE, dtype=np.float32)
    while True:
        try:
            data = stream.read(1024, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = np.roll(audio_buffer, -len(samples)); audio_buffer[-len(samples):] = samples
            interpreter.set_tensor(input_details[0]['index'], audio_buffer)
            interpreter.invoke()
            if interpreter.get_tensor(output_details[0]['index']).flatten()[COUGH_INDEX] > 0.3:
                status["cough_latched"] = True
                # FIXED: Send alert and then sleep for 3 seconds to prevent spam
                send_to_telegram(f"🚨 COUGH DETECTED! Temp: {status['live_temp']}°C")
                time.sleep(3) 
        except: pass

# --- 6. AUTO REASONING ENGINE ---
def run_diagnosis(animal, temp, cough, lethargy):
    try:
        df = pd.read_csv(f"{animal}_dataset.csv")
        df.columns = df.columns.str.strip()
        df['Body_Temperature'] = pd.to_numeric(df['Body_Temperature'].astype(str).str.replace('°C', '', regex=False).str.strip(), errors='coerce')
        df = df.dropna(subset=['Body_Temperature'])
        matches = df[(df['Body_Temperature'] >= (temp - 1.0)) & (df['Body_Temperature'] <= (temp + 1.0))].copy()
        
        def score_row(row):
            score = 0
            val = str(row['Coughing']).strip().lower()
            if (val in ['yes', '1', 'true']) == cough: score += 2
            syms = " ".join([str(row[c]) for c in df.columns if 'Symptom' in c]).lower()
            if lethargy and 'letharg' in syms: score += 2
            return score

        matches['Score'] = matches.apply(score_row, axis=1)
        matches = matches[matches['Score'] > 0].sort_values(by='Score', ascending=False)
        
        if matches.empty: return "   No clinical matches found."
        
        results = matches['Disease_Prediction'].value_counts()
        report = ""
        for disease, count in results.items():
            prob = (count / len(matches)) * 100
            if prob < 5: continue 
            report += f"   → {disease}: {prob:.1f}% Match\n"
            
            disease_rows = matches[matches['Disease_Prediction'] == disease]
            symptom_cols = [c for c in df.columns if 'Symptom' in c]
            all_syms = []
            for _, row in disease_rows.iterrows():
                all_syms.extend([str(row[c]) for c in symptom_cols if pd.notna(row[c]) and str(row[c]).lower() != 'nan'])
            unique_syms = list(set([s for s in all_syms if 'letharg' not in s.lower()]))[:2]
            if unique_syms:
                report += f"     (Note: Check for {', '.join(unique_syms)})\n"
        return report
    except: return "   Error reading dataset."

# --- 7. AUTOMATED DASHBOARD ---
threading.Thread(target=sensor_loop, daemon=True).start()
threading.Thread(target=audio_loop, daemon=True).start()

print("\n🚀 AUTONOMOUS AI ACTIVE: Mirroring results to Telegram.")

try:
    while True:
        t = status["live_temp"]
        c = status["cough_latched"]
        l = status["act"] < LETHARGY_THRESHOLD
        
        header = f"⏰ {time.strftime('%H:%M:%S')} | Temp={t}°C | Cough={c} | Lethargy={l}"
        cow_diag = run_diagnosis("cow", t, c, l)
        goat_diag = run_diagnosis("goat", t, c, l)
        
        full_report = f"{header}\n\n🐄 COW ANALYSIS:\n{cow_diag}\n🐐 GOAT ANALYSIS:\n{goat_diag}"
        
        print("-" * 50)
        print(full_report)
        print("-" * 50 + "\n")
        
        send_to_telegram(full_report)
        
        status["cough_latched"] = False
        time.sleep(10)

except KeyboardInterrupt:
    print("\nShutting down...")
