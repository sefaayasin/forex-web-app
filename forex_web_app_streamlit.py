"""Compatibility entry point for existing Streamlit deployments."""
from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name("forex_web_app_streamlit_v14_alert_decision.py")),
    run_name="__main__",
)
