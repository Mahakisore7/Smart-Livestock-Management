import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Herd Sentinel", layout="wide")

st.markdown("""
    <style>
    .report-box { background-color:#e1f5fe; padding:20px; border-radius:15px; border-left: 5px solid #0288d1; color:#01579b; font-size:18px; }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    conn = sqlite3.connect('livestock_agent.db')
    df = pd.read_sql_query("SELECT * FROM health_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

st.title("🏥 Herd Health Command Center")

try:
    df = load_data()
    if not df.empty:
        # Sidebar Selection
        available_cows = df['cow_id'].unique()
        selected_cow = st.sidebar.selectbox("Select Animal ID", available_cows)
        
        cow_df = df[df['cow_id'] == selected_cow]
        latest = cow_df.iloc[0]

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Body Temperature", f"{latest['avg_temp']}°C")
        c2.metric("Coughs (Last Min)", latest['cough_count'])
        c3.metric("Current Match", latest['diagnosis'])

        # Agent Advice
        st.subheader("🤖 Veterinary Agent Alert")
        st.markdown(f'<div class="report-box">{latest["agent_advice"]}</div>', unsafe_allow_html=True)

        # Graph
        st.subheader("📈 Clinical Trends")
        fig = px.line(cow_df, x='timestamp', y='avg_temp', markers=True, title=f"Temperature History: {selected_cow}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data...")
except:
    st.error("Connect to the Agent script first.")
