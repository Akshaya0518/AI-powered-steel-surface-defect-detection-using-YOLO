
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import json
import time

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="DefectIQ | Steel Inspection",
    page_icon="🔍",
    layout="wide"
)

# Dashboard styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 1.5rem;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e1e6ef;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🔍 DefectIQ")
st.sidebar.caption("AI-Powered Steel Surface Inspection")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Inspect Image",
        "Batch Inspection",
        "Quality Analytics",
        "Reports",
        "Model Health"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Steel Surface Defect Detection")


def load_image(uploaded_file):
    """Convert uploaded image to OpenCV BGR format."""
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb, image_bgr


def show_result(result):
    """Display a result returned by the inspection pipeline."""
    if not isinstance(result, dict):
        st.warning("The inspection returned an unexpected result format.")
        st.write(result)
        return

    st.subheader("Inspection Result")

    decision = str(result.get("decision", "UNKNOWN")).upper()
    reason = result.get("reason", "No reason provided.")

    if decision == "ACCEPT":
        st.success(f"Decision: {decision}")
    elif decision in ("REJECT", "FAIL"):
        st.error(f"Decision: {decision}")
    else:
        st.warning(f"Decision: {decision}")

    st.write("**Reason:**", reason)

    defects = result.get("defects", [])
    if isinstance(defects, list):
        st.write("**Defects detected:**", len(defects))

        if defects:
            st.dataframe(pd.json_normalize(defects), use_container_width=True)

    with st.expander("View complete inspection data"):
        st.json(result)


def inspect_uploaded_image(image_bgr, filename):
    """Call the project's existing pipeline."""
    try:
        from pipeline import inspect

        return inspect(
            image_bgr,
            filename,
            save=True
        )
    except Exception as error:
        st.error("The project inspection pipeline could not complete.")
        st.exception(error)
        return None


# OVERVIEW
if page == "Overview":
    st.title("Steel Surface Inspection Dashboard")
    st.write(
        "Inspect steel sheets, identify surface defects, "
        "and review quality inspection results."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Inspection System", "Ready to test")
    col2.metric("Inspection Mode", "AI-assisted")
    col3.metric("Input Format", "Image")

    st.divider()
    st.subheader("System Workflow")

    st.markdown("""
    1. Upload a steel surface image.
    2. Detect scratches and other surface defects.
    3. Evaluate defect severity.
    4. Generate an inspection decision.
    5. Review and record the inspection result.
    """)

    st.info(
        "Upload an image in the Inspect Image section to test "
        "the connected inspection pipeline."
    )


# SINGLE IMAGE INSPECTION
elif page == "Inspect Image":
    st.title("Inspect Steel Surface")

    uploaded_file = st.file_uploader(
        "Upload a steel surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="single_image"
    )

    if uploaded_file is not None:
        image_rgb, image_bgr = load_image(uploaded_file)

        st.image(
            image_rgb,
            caption="Uploaded steel surface",
            use_container_width=True
        )

        if st.button("Run Inspection", type="primary"):
            with st.spinner("Inspecting image..."):
                result = inspect_uploaded_image(
                    image_bgr,
                    uploaded_file.name
                )

            if result is not None:
                show_result(result)


# BATCH INSPECTION
elif page == "Batch Inspection":
    st.title("Batch Image Inspection")

    uploaded_files = st.file_uploader(
        "Upload multiple steel surface images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_images"
    )

    if uploaded_files and st.button("Inspect All Images", type="primary"):
        results = []

        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            _, image_bgr = load_image(uploaded_file)

            with st.spinner(f"Inspecting {uploaded_file.name}..."):
                result = inspect_uploaded_image(
                    image_bgr,
                    uploaded_file.name
                )

            if result is not None:
                results.append({
                    "Filename": uploaded_file.name,
                    "Decision": result.get("decision", "UNKNOWN"),
                    "Reason": result.get("reason", ""),
                    "Defect Count": len(result.get("defects", []))
                    if isinstance(result.get("defects", []), list)
                    else 0
                })

            progress.progress((index + 1) / len(uploaded_files))

        if results:
            st.subheader("Batch Results")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download Results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="inspection_results.csv",
                mime="text/csv"
            )


# QUALITY ANALYTICS
elif page == "Quality Analytics":
    st.title("Quality Analytics")

    st.write(
        "Analytics will be populated as inspection results "
        "are recorded by the backend."
    )

    st.info(
        "Connect this section to saved inspection records "
        "to display real defect counts and acceptance rates."
    )


# REPORTS
elif page == "Reports":
    st.title("Inspection Reports")

    st.write(
        "Run an inspection from the Inspect Image or Batch "
        "Inspection section to generate downloadable results."
    )

    st.info(
        "This page can be connected to the project's database "
        "for historical reports."
    )


# MODEL HEALTH
elif page == "Model Health":
    st.title("Model Health")

    st.write("Basic environment checks")

    col1, col2 = st.columns(2)

    col1.metric("OpenCV", cv2.__version__)
    col2.metric("Python", sys.version.split()[0])

    model_path = BASE_DIR / "models" / "best.pt"

    if model_path.exists():
        st.success("Model file found: models/best.pt")
    else:
        st.warning(
            "models/best.pt was not found. "
            "Confirm the trained model location with Member A."
        )

    try:
        import ultralytics
        st.success(
            f"Ultralytics installed: {ultralytics.__version__}"
        )
    except ImportError:
        st.warning(
            "Ultralytics is not installed in this Python environment."
        )
