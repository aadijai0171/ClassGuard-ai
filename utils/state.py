"""
state.py
────────
Initialise and manage Streamlit session-state singletons.
"""

import streamlit as st
from utils.detection_engine import DetectionEngine
from utils.alert_manager import AlertManager


def init_state():
    """Call once at the top of every page to ensure state is bootstrapped."""

    if "engine" not in st.session_state:
        st.session_state.engine = DetectionEngine()

    if "alert_mgr" not in st.session_state:
        st.session_state.alert_mgr = AlertManager()

    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False

    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []      # list of dicts for UI display

    if "history" not in st.session_state:
        # rolling list of DetectionResult objects (last 300 = ~5 min @ 1fps)
        st.session_state.history = []

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "teacher_name": "",
            "teacher_email": "",
            "coordinator_name": "",
            "coordinator_email": "",
            "principal_email": "",
            "chaos_trigger": 50,
            "teacher_timeout": 60,
            "coord_timeout": 120,
            "sensitivity": 50,
            "camera_index": 0,
            "yolo_model": "yolov8n.pt",
            "show_heatmap": True,
        }


def get_engine() -> DetectionEngine:
    init_state()
    return st.session_state.engine


def get_alert_mgr() -> AlertManager:
    init_state()
    return st.session_state.alert_mgr