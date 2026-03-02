from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app)

DB_PATH = os.path.expanduser("/home/d11/livestock_ai/livestock_agent/livestock_app_v2.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/active_cow', methods=['GET'])
def get_active_cow():
    try:
        conn = get_db_connection()
        row = conn.execute('SELECT cow_id FROM health_logs ORDER BY id DESC LIMIT 1').fetchone()
        conn.close()
        return jsonify({"cow_id": row['cow_id'] if row else ""})
    except:
        return jsonify({"cow_id": ""})

@app.route('/cows', methods=['GET'])
def get_cows():
    try:
        conn = get_db_connection()
        rows = conn.execute('SELECT DISTINCT cow_id FROM health_logs').fetchall()
        conn.close()
        return jsonify([row['cow_id'] for row in rows])
    except:
        return jsonify([])

@app.route('/latest/<cow_id>', methods=['GET'])
def get_latest(cow_id):
    try:
        conn = get_db_connection()
        log = conn.execute('SELECT * FROM health_logs WHERE cow_id = ? ORDER BY id DESC LIMIT 1', (cow_id,)).fetchone()
        history = conn.execute('SELECT avg_temp FROM health_logs WHERE cow_id = ? ORDER BY id DESC LIMIT 20', (cow_id,)).fetchall()
        conn.close()

        if log:
            return jsonify({
                "id": log['cow_id'],
                "animal": log['animal'] if 'animal' in log.keys() else "Cow", # 🟢 NEW: Sends Species Info
                "temp": log['avg_temp'],
                "coughs": log['cough_count'],
                "activity": log['activity_level'],
                "diagnosis": log['diagnosis'],
                "advice": log['agent_advice'],
                "temp_history": [row['avg_temp'] for row in reversed(history)],
                "server_time": os.popen("date +%H:%M:%S").read().strip()
            })
        return jsonify({"error": "No records"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history/<cow_id>', methods=['GET'])
def get_history(cow_id):
    try:
        conn = get_db_connection()
        rows = conn.execute('SELECT * FROM health_logs WHERE cow_id = ? ORDER BY id DESC', (cow_id,)).fetchall()
        conn.close()
        
        result = []
        for r in rows:
            d = dict(r)
            if 'timestamp' not in d: d['timestamp'] = "Recent"
            result.append(d)
        return jsonify(result)
    except Exception as e:
        return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
