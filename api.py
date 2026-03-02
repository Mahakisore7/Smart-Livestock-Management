from flask import Flask, jsonify
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app) # 🟢 Allows your phone to connect without security blocks

# 📍 Absolute path to your database
DB_PATH = "/home/d11/livestock_ai/livestock_agent/livestock_app_v2.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/cows', methods=['GET'])
def get_cows():
    """Returns a unique list of all Cow IDs found in the database."""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify([])
        conn = get_db_connection()
        rows = conn.execute('SELECT DISTINCT cow_id FROM health_logs').fetchall()
        conn.close()
        return jsonify([row['cow_id'] for row in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/latest/<cow_id>', methods=['GET'])
def get_latest(cow_id):
    """Returns the most recent health record for a specific Cow ID."""
    try:
        if not os.path.exists(DB_PATH):
            return jsonify({"error": "Database not found"}), 404
        
        conn = get_db_connection()
        log = conn.execute(
            'SELECT * FROM health_logs WHERE cow_id = ? ORDER BY id DESC LIMIT 1', 
            (cow_id,)
        ).fetchone()
        conn.close()

        if log:
            return jsonify({
                "id": log['cow_id'],
                "temp": log['avg_temp'],
                "coughs": log['cough_count'],
                "activity": log['activity_level'],
                "diagnosis": log['diagnosis'],
                "advice": log['agent_advice']
            })
        return jsonify({"error": "No records found for this ID"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Listen on 0.0.0.0 so the phone can reach the Pi
    app.run(host='0.0.0.0', port=5000, debug=False)
