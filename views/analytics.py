"""
views/analytics.py  –  Analytics Dashboard
"""

import time
import numpy as np
import streamlit as st
from utils.state import init_state


def show():
    init_state()
    history = st.session_state.history

    st.title("📊 Analytics Dashboard")

    if not history:
        st.info("No monitoring data yet. Start the **Live Monitor** first.")
        return

    teacher_present_pct = np.mean([r.teacher_present for r in history]) * 100
    avg_chaos = np.mean([r.chaos_score for r in history])
    avg_motion = np.mean([r.motion_level for r in history])
    avg_noise = np.mean([r.noise_level for r in history])
    avg_persons = np.mean([r.person_count for r in history])
    total_alerts = len(st.session_state.alert_log)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Teacher Present", f"{teacher_present_pct:.0f}%")
    c2.metric("Avg Chaos", f"{avg_chaos:.0f}%")
    c3.metric("Avg Motion", f"{avg_motion:.0f}%")
    c4.metric("Avg Noise", f"{avg_noise:.0f}%")
    c5.metric("Avg Persons", f"{avg_persons:.1f}")
    c6.metric("Total Alerts", total_alerts)

    st.divider()

    import pandas as pd

    records = [
        {
            "time": i,
            "Chaos Score": r.chaos_score,
            "Motion Level": r.motion_level,
            "Noise Level": r.noise_level,
            "Persons": r.person_count,
            "Teacher Present": int(r.teacher_present) * 100,
        }
        for i, r in enumerate(history)
    ]
    df = pd.DataFrame(records)

    st.markdown("### Chaos / Motion / Noise Over Time")
    st.line_chart(df.set_index("time")[["Chaos Score", "Motion Level", "Noise Level"]])

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Person Count Over Time")
        st.area_chart(df.set_index("time")[["Persons"]])
    with col_r:
        st.markdown("### Teacher Presence (% of time)")
        st.bar_chart(df.set_index("time")[["Teacher Present"]])

    st.divider()

    st.markdown("### Alert History")
    alerts = st.session_state.alert_log
    if alerts:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df, use_container_width=True)
        csv = alert_df.to_csv(index=False)
        st.download_button(
            "⬇ Download Alert Log (CSV)",
            data=csv,
            file_name=f"alert_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No alerts triggered during this session.")

    st.divider()

    st.markdown("### Chaos Score Distribution")
    chaos_vals = [r.chaos_score for r in history]
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    counts, _ = np.histogram(chaos_vals, bins=bins)
    dist_df = pd.DataFrame({"Range": labels, "Frames": counts}).set_index("Range")
    st.bar_chart(dist_df)
