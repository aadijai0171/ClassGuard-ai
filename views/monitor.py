"""
views/monitor.py  –  Live Classroom Monitor
"""

import time
import cv2
import numpy as np
import streamlit as st
from utils.state import init_state, get_engine, get_alert_mgr
from utils.alert_manager import AlertLevel


def show():
    init_state()
    engine = get_engine()
    alert_mgr = get_alert_mgr()

    st.title("📹 Live Classroom Monitor")

    col_a, col_b, col_c = st.columns([2, 2, 3])
    with col_a:
        cam_idx = st.session_state.settings.get("camera_index", 0)
        source = st.selectbox(
            "Video source",
            ["Webcam (default)", "Upload video file", "Demo mode"],
            index=2,
        )
    with col_b:
        sensitivity = st.slider(
            "Motion sensitivity", 10, 100,
            st.session_state.settings.get("sensitivity", 50), 5,
        )
        st.session_state.settings["sensitivity"] = sensitivity
    with col_c:
        start_col, stop_col, ack_col = st.columns(3)
        with start_col:
            if st.button("▶ Start", width="stretch", type="primary"):
                st.session_state.monitoring_active = True
        with stop_col:
            if st.button("⏹ Stop", width="stretch"):
                st.session_state.monitoring_active = False
        with ack_col:
            if st.button("✅ Ack Alert", width="stretch"):
                alert_mgr.acknowledge()
                st.success("Alert acknowledged")

    st.divider()

    vid_col, stat_col = st.columns([3, 1])

    with vid_col:
        frame_placeholder = st.empty()
        uploaded_video = None
        if source == "Upload video file":
            uploaded_video = st.file_uploader("Upload an MP4 / AVI file", type=["mp4", "avi", "mov"])

    with stat_col:
        st.markdown("### Live Metrics")
        metric_placeholder = st.empty()
        alert_placeholder = st.empty()

    st.markdown("### 🔔 Alert Log")
    log_placeholder = st.empty()

    if not st.session_state.monitoring_active:
        frame_placeholder.info("Press **▶ Start** to begin monitoring.")
        _render_log(log_placeholder)
        return

    cap = _open_source(source, cam_idx, uploaded_video)
    if cap is None:
        st.error("Could not open video source. Switching to demo mode.")
        cap = _DemoCapture()

    fps_target = 15
    frame_time = 1.0 / fps_target

    try:
        while st.session_state.monitoring_active:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                if source == "Demo mode":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            result = engine.process_frame(frame, sensitivity=sensitivity)

            st.session_state.history.append(result)
            if len(st.session_state.history) > 300:
                st.session_state.history.pop(0)

            alert_event = alert_mgr.update(result.teacher_present, result.chaos_score)
            if alert_event:
                st.session_state.alert_log.append({
                    "time": time.strftime("%H:%M:%S", time.localtime(alert_event.timestamp)),
                    "level": alert_event.level.name,
                    "chaos": f"{alert_event.chaos_score:.0f}%",
                    "msg": alert_event.message,
                    "ack": alert_event.acknowledged,
                })

            if result.annotated_frame is not None:
                rgb = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", width="stretch")

            absence_s = int(alert_mgr.get_elapsed_absence())
            level = alert_mgr.get_current_level()
            badge_cls = {"NONE": "badge-green", "TEACHER": "badge-amber",
                         "COORDINATOR": "badge-red", "CRITICAL": "badge-red"}.get(level.name, "badge-green")

            metric_placeholder.markdown(f"""
<div class="metric-card {'red' if not result.teacher_present else 'green'}">
  <b>Teacher</b><br>
  <span class="badge {'badge-green' if result.teacher_present else 'badge-red'}">
    {'Present' if result.teacher_present else 'ABSENT'}
  </span>
  {'<br><small>' + result.teacher_name + ' (' + f"{result.teacher_confidence:.0%}" + ')</small>' if result.teacher_name else ''}
</div>
<div class="metric-card {'red' if result.chaos_score > 70 else 'amber' if result.chaos_score > 40 else 'green'}">
  <b>Chaos Score</b><br>
  <span style="font-size:28px;font-weight:700;color:#0f172a">{result.chaos_score:.0f}%</span>
</div>
<div class="metric-card">
  <b>People</b><br>
  <span style="font-size:24px;font-weight:700">{result.person_count}</span>
</div>
<div class="metric-card">
  <b>Motion</b><br>
  <div style="background:#e2e8f0;border-radius:4px;height:8px;width:100%">
    <div style="background:#3b82f6;width:{min(result.motion_level,100):.0f}%;height:8px;border-radius:4px"></div>
  </div>
  <small>{result.motion_level:.0f}%</small>
</div>
<div class="metric-card">
  <b>Noise</b><br>
  <div style="background:#e2e8f0;border-radius:4px;height:8px;width:100%">
    <div style="background:#8b5cf6;width:{min(result.noise_level,100):.0f}%;height:8px;border-radius:4px"></div>
  </div>
  <small>{result.noise_level:.0f}%</small>
</div>
<div class="metric-card">
  <b>Groups detected</b><br>
  <span style="font-size:20px">{result.group_formations}</span>
</div>
<div class="metric-card {'red' if level != AlertLevel.NONE else ''}">
  <b>Alert level</b><br>
  <span class="badge {badge_cls}">{level.name}</span>
  <br><small>Absence: {absence_s}s</small>
</div>
""", unsafe_allow_html=True)

            if alert_event:
                alert_placeholder.error(alert_event.message)

            _render_log(log_placeholder)

            elapsed = time.time() - t0
            wait = frame_time - elapsed
            if wait > 0:
                time.sleep(wait)

    finally:
        cap.release()
        st.session_state.monitoring_active = False


def _open_source(source: str, cam_idx: int, uploaded_file):
    if source == "Webcam (default)":
        cap = cv2.VideoCapture(cam_idx)
        return cap if cap.isOpened() else None
    elif source == "Upload video file" and uploaded_file:
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
        return cap if cap.isOpened() else None
    else:
        return _DemoCapture()


def _render_log(placeholder):
    logs = st.session_state.alert_log[-10:][::-1]
    if not logs:
        placeholder.info("No alerts triggered yet.")
        return
    rows = ""
    for log in logs:
        color = {"TEACHER": "#fef3c7", "COORDINATOR": "#fee2e2",
                 "CRITICAL": "#fee2e2", "NONE": "#f0fdf4"}.get(log["level"], "#f8fafc")
        rows += f"""
        <tr style="background:{color}">
          <td style="padding:6px 10px">{log['time']}</td>
          <td style="padding:6px 10px"><b>{log['level']}</b></td>
          <td style="padding:6px 10px">{log['chaos']}</td>
          <td style="padding:6px 10px;font-size:13px">{log['msg'][:80]}…</td>
        </tr>"""
    placeholder.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:14px">
  <thead>
    <tr style="background:#f1f5f9">
      <th style="padding:6px 10px;text-align:left">Time</th>
      <th style="padding:6px 10px;text-align:left">Level</th>
      <th style="padding:6px 10px;text-align:left">Chaos</th>
      <th style="padding:6px 10px;text-align:left">Message</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)


class _DemoCapture:
    def __init__(self):
        self._t = 0

    def read(self):
        self._t += 1
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (245, 240, 230)
        cv2.rectangle(frame, (80, 30), (560, 180), (30, 80, 50), -1)
        cv2.putText(frame, "CLASSROOM DEMO", (150, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 230, 200), 2)
        for i in range(5):
            x = 60 + i * 110 + int(8 * np.sin(self._t / 10 + i))
            y = 300 + int(5 * np.cos(self._t / 8 + i * 1.3))
            cv2.ellipse(frame, (x, y - 20), (18, 20), 0, 0, 360, (200, 160, 100), -1)
            cv2.rectangle(frame, (x - 18, y), (x + 18, y + 60),
                          (60 + i * 20, 80, 200 - i * 15), -1)
        time.sleep(0.03)
        return True, frame

    def set(self, prop, val):
        self._t = 0

    def release(self):
        pass
