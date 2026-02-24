import sqlite3

def init_db():
    conn = sqlite3.connect('livestock_agent.db')
    c = conn.cursor()
    # This table stores every 10-minute pulse from the sensors
    c.execute('''CREATE TABLE IF NOT EXISTS health_logs
                 (timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                  animal TEXT, 
                  avg_temp REAL, 
                  cough_count INTEGER, 
                  activity_level TEXT, 
                  diagnosis TEXT, 
                  agent_advice TEXT)''')
    conn.commit()
    conn.close()
    print("✅ Database Memory Created Successfully!")

if __name__ == "__main__":
    init_db()
