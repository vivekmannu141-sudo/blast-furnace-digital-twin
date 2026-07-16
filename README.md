# Blast Furnace Historian Replay Prototype

This package is a realistic offline prototype using synthetic data. It does not connect to a plant system.

## Run on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501.

## Included
- 7 days of one-minute synthetic historian data
- Stable, cold-furnace, hanging-risk, hot-furnace, and sensor-degradation periods
- Tag dictionary
- CSV upload validation
- Historian replay slider
- Trends, data-quality analysis, alerts, and what-if sandbox

## Safety
Simulation/advisory only. No PLC, DCS, SCADA, OPC UA, historian, or write-back interface is included. All limits, equations, alerts, and models require validation by authorized plant experts.
# Blast Furnace Digital Twin

An offline Streamlit prototype demonstrating a Blast Furnace Digital
Twin using synthetic historian-style process data.

## Features

- Historical replay
- Furnace-state visualization
- Process trends
- Burden and fuel monitoring
- Gas and pressure monitoring
- Cooling monitoring
- Advisory predictions
- Alarm console
- Scenario analysis
- Data-quality assessment
- Engineering analytics
- Shift reporting

## Run locally

Create or activate a Python virtual environment and install dependencies:

    python -m pip install -r requirements.txt

Start the application:

    python -m streamlit run app.py

## Data

The included dataset is synthetic and supplied only for software
demonstration and engineering prototyping.

## Important limitation

The models, thresholds, calculations, predictions and scenarios are not
validated for plant operation, equipment protection, safety decisions
or process control. The application has no control or write-back
capability.