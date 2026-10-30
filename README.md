# AGENT X - Campus Energy & Traffic Optimizer
pip install -r requirements.txt
streamlit run app.py
- No CSV? App uses synthetic demo data. Upload gate_traffic.csv / power_logs.csv in the sidebar to use real data.
- Tick "Use Claude" + paste Anthropic API key for the LLM tool-calling agent (otherwise rule-based agent runs).
- Everything is a SIMULATION on proxy data (simulated timetable, rooms, 4 kW/room, etc.). Edit assumptions in sidebar.
