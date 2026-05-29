"""
views/enrollment.py  –  Teacher Enrollment
"""

import cv2
import numpy as np
import streamlit as st
from utils.state import init_state, get_engine


def show():
    init_state()
    engine = get_engine()

    st.title("🧑‍🏫 Teacher Enrollment")
    st.markdown("Register teachers so the system can recognise them automatically in the live feed.")

    st.markdown("### Enrolled Teachers")
    enrolled = engine.enrolled_teachers
    if enrolled:
        for name in enrolled:
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"✅ **{name}**")
            if c2.button("Remove", key=f"remove_{name}"):
                engine.remove_teacher(name)
                st.rerun()
    else:
        st.info("No teachers enrolled yet. Add one below.")

    st.divider()

    st.markdown("### Enroll New Teacher")
    tab_upload, tab_webcam = st.tabs(["📁 Upload Photo", "📷 Capture from Webcam"])

    with tab_upload:
        name_u = st.text_input("Teacher name", key="enroll_name_upload")
        photo = st.file_uploader(
            "Upload a clear, front-facing photo (.jpg / .png)",
            type=["jpg", "jpeg", "png"],
            key="enroll_photo",
        )
        if photo and name_u:
            file_bytes = np.frombuffer(photo.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Preview", width=240)

            if st.button("✅ Enroll Teacher", type="primary"):
                ok = engine.enroll_teacher(img, name_u)
                if ok:
                    st.success(f"**{name_u}** enrolled successfully!")
                else:
                    st.error("No face detected in the photo. Please use a clear, front-facing image.")

    with tab_webcam:
        st.info("Webcam capture is not available on Streamlit Cloud. Please upload a photo instead.")

    st.divider()

    with st.expander("📌 Tips for best recognition accuracy"):
        st.markdown("""
- Use a **well-lit, front-facing** photo — avoid sunglasses or hats.
- Enroll **multiple photos** of the same teacher (different lighting) for better robustness.
- Face recognition uses **DeepFace (Facenet)** — no dlib compilation needed.
        """)
