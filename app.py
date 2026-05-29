"""
AI-Powered Automated Teacher Presence Monitoring System
Main entry point
"""

import streamlit as st

st.set_page_config(
    page_title="ClassGuard AI",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* sidebar */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stRadio label { font-size: 15px; }

  /* main area */
  .main { background: #f8fafc; }

  /* metric cards */
  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    border-left: 4px solid #3b82f6;
    margin-bottom: 12px;
  }
  .metric-card.red   { border-left-color: #ef4444; }
  .metric-card.green { border-left-color: #22c55e; }
  .metric-card.amber { border-left-color: #f59e0b; }

  /* status badge */
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
  }
  .badge-green { background:#dcfce7; color:#166534; }
  .badge-red   { background:#fee2e2; color:#991b1b; }
  .badge-amber { background:#fef3c7; color:#92400e; }

  div[data-testid="stAlert"] { border-radius: 10px; }
  h1, h2, h3 { color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── sidebar nav ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏫 ClassGuard AI")
    st.markdown("*Teacher Presence Monitoring*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📹 Live Monitor",
            "🧑‍🏫 Teacher Enrollment",
            "📊 Analytics Dashboard",
            "⚙️ Settings",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("v1.0  •  ClassGuard AI")

# ── page routing ─────────────────────────────────────────────────────────────
if page == "📹 Live Monitor":
    from views import monitor
    monitor.show()

elif page == "🧑‍🏫 Teacher Enrollment":
    from views import enrollment
    enrollment.show()

elif page == "📊 Analytics Dashboard":
    from views import analytics
    analytics.show()

elif page == "⚙️ Settings":
    from views import settings
    settings.show()

else:
    from views import about
    about.show()