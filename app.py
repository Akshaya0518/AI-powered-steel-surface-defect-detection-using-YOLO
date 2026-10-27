
import sys
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# =========================================================
# PROJECT PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DefectIQ | Steel Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DASHBOARD STYLING — HIGH CONTRAST
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
    color: #172033;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label,
.stApp span,
.stApp li {
    color: #253047 !important;
}

[data-testid="stHeader"] {
    background-color: #f5f7fb;
}

[data-testid="stSidebar"] {
    background-color: #202638;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px;
}

.stButton > button {
    background-color: #2563eb;
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    padding: 10px 18px;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)
# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
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
        "Model Health",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Steel Surface Defect Detection")


# =========================================================
# PIPELINE AND STORAGE
# =========================================================
def get_pipeline():
    """Import Member B's existing inspection pipeline."""
    from src.pipeline import inspect
    return inspect


def get_storage():
    """Import Member B's existing SQLite storage functions."""
    from src.storage import load_history, defect_counts
    return load_history, defect_counts


def load_image(uploaded_file):
    """Convert an uploaded image to RGB and OpenCV BGR formats."""
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_rgb, image_bgr


def run_inspection(image_bgr, filename):
    """Run the existing inspection pipeline and save the result."""
    try:
        inspect = get_pipeline()
        return inspect(image_bgr, filename, save=True)
    except Exception as error:
        st.error("The inspection pipeline could not complete.")
        st.exception(error)
        return None


def get_history():
    """Read saved inspection history."""
    try:
        load_history, _ = get_storage()
        return load_history()
    except Exception as error:
        st.error("Could not load inspection history.")
        st.exception(error)
        return None


def get_defect_counts():
    """Read defect statistics from storage."""
    try:
        _, defect_counts = get_storage()
        return defect_counts()
    except Exception as error:
        st.error("Could not load defect statistics.")
        st.exception(error)
        return None


# =========================================================
# DISPLAY INSPECTION RESULT
# =========================================================
def show_result(result):
    if not isinstance(result, dict):
        st.warning("The pipeline returned an unexpected result.")
        st.write(result)
        return

    decision = str(result.get("decision", "UNKNOWN")).upper()
    reason = result.get("reason", "No reason provided.")
    defects = result.get("defects", [])
    defects = defects if isinstance(defects, list) else []

    st.subheader("Inspection Result")

    if decision == "ACCEPT":
        st.success(f"Decision: {decision}")
    elif decision == "REJECT":
        st.error(f"Decision: {decision}")
    else:
        st.warning(f"Decision: {decision}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Decision", decision)
    col2.metric("Defects Detected", len(defects))

    inference_ms = result.get("inference_ms")
    if isinstance(inference_ms, (int, float)):
        col3.metric("Inference Time", f"{inference_ms:.1f} ms")
    else:
        col3.metric("Inference Time", "N/A")

    st.write("**Reason:**", reason)

    annotated = result.get("annotated")

    if isinstance(annotated, np.ndarray):
        try:
            annotated_rgb = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )
            st.image(
                annotated_rgb,
                caption="Annotated inspection image",
                use_container_width=True,
            )
        except cv2.error:
            st.warning("Could not display the annotated image.")

    if defects:
        st.subheader("Detected Defects")
        st.dataframe(
            pd.json_normalize(defects),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No defects were returned by the detector.")

    with st.expander("View complete inspection data"):
        safe_result = {
            key: value
            for key, value in result.items()
            if key != "annotated"
        }
        st.json(safe_result)


def show_table(data):
    """Display tabular results when records exist."""
    if data is None:
        st.warning("Data could not be loaded.")
    elif isinstance(data, pd.DataFrame) and not data.empty:
        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No saved records are available yet.")


# =========================================================
# 1. OVERVIEW
# =========================================================
if page == "Overview":
    st.title("Steel Surface Inspection Dashboard")

    st.write(
        "Inspect steel sheets, identify surface defects, "
        "evaluate severity, and review inspection results."
    )

    history = get_history()

    total_inspections = (
        len(history)
        if isinstance(history, pd.DataFrame)
        else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Saved Inspections", total_inspections)
    col2.metric("Inspection Mode", "AI-assisted")
    col3.metric("Input Format", "Image")

    st.divider()
    st.subheader("Inspection Workflow")

    st.markdown(
        """
        1. Upload a steel surface image.
        2. Member A's detector identifies possible defects.
        3. Member B's severity engine evaluates defect severity.
        4. The pipeline produces an inspection decision.
        5. Results are saved for history and reporting.
        """
    )

    st.info(
        "Select **Inspect Image** from the sidebar to begin."
    )


# =========================================================
# 2. SINGLE IMAGE INSPECTION
# =========================================================
elif page == "Inspect Image":
    st.title("Inspect Steel Surface")

    uploaded_file = st.file_uploader(
        "Upload a steel surface image",
        type=["jpg", "jpeg", "png", "bmp"],
        key="single_image",
    )

    if uploaded_file is not None:
        try:
            image_rgb, image_bgr = load_image(uploaded_file)

            st.image(
                image_rgb,
                caption="Uploaded steel surface",
                use_container_width=True,
            )

            if st.button("Run Inspection", type="primary"):
                with st.spinner("Inspecting image..."):
                    result = run_inspection(
                        image_bgr,
                        uploaded_file.name,
                    )

                if result is not None:
                    show_result(result)

        except Exception as error:
            st.error("Could not read the uploaded image.")
            st.exception(error)


# =========================================================
# 3. BATCH INSPECTION
# =========================================================
elif page == "Batch Inspection":
    st.title("Batch Image Inspection")

    uploaded_files = st.file_uploader(
        "Upload multiple steel surface images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_images",
    )

    if uploaded_files and st.button(
        "Inspect All Images",
        type="primary",
    ):
        results = []
        progress = st.progress(0.0)

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                _, image_bgr = load_image(uploaded_file)

                with st.spinner(
                    f"Inspecting {uploaded_file.name}..."
                ):
                    result = run_inspection(
                        image_bgr,
                        uploaded_file.name,
                    )

                if isinstance(result, dict):
                    defects = result.get("defects", [])
                    if not isinstance(defects, list):
                        defects = []

                    results.append(
                        {
                            "Filename": uploaded_file.name,
                            "Decision": result.get(
                                "decision", "UNKNOWN"
                            ),
                            "Reason": result.get("reason", ""),
                            "Defect Count": len(defects),
                            "Inference Time (ms)": result.get(
                                "inference_ms"
                            ),
                            "Database ID": result.get("id"),
                        }
                    )

            except Exception as error:
                st.error(
                    f"Could not inspect {uploaded_file.name}: {error}"
                )

            progress.progress((index + 1) / len(uploaded_files))

        if results:
            st.subheader("Batch Results")
            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Batch Results CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="inspection_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("No images were successfully inspected.")


# =========================================================
# 4. QUALITY ANALYTICS
# =========================================================
elif page == "Quality Analytics":
    st.title("Quality Analytics")
    st.caption("Analytics based on records returned by storage.")

    history = get_history()
    counts = get_defect_counts()

    if isinstance(history, pd.DataFrame) and not history.empty:
        st.subheader("Inspection Summary")

        col1, col2 = st.columns(2)
        col1.metric("Total Inspections", len(history))

        if "decision" in history.columns:
            rejected = (
                history["decision"]
                .astype(str)
                .str.upper()
                .eq("REJECT")
                .sum()
            )
            col2.metric("Rejected Inspections", int(rejected))
        else:
            col2.metric("Rejected Inspections", "N/A")

        st.subheader("Inspection History")
        show_table(history)

        if "decision" in history.columns:
            decisions = (
                history["decision"]
                .fillna("UNKNOWN")
                .astype(str)
                .value_counts()
            )

            st.subheader("Decisions by Category")
            st.bar_chart(decisions)
    else:
        st.info(
            "No saved inspection history is available. "
            "Inspect an image first."
        )

    st.divider()
    st.subheader("Defect Analytics")

    if isinstance(counts, pd.DataFrame) and not counts.empty:
        show_table(counts)

        if "type" in counts.columns:
            defect_summary = (
                counts["type"]
                .fillna("Unknown")
                .value_counts()
            )

            st.subheader("Defects by Type")
            st.bar_chart(defect_summary)

        if "decision" in counts.columns:
            decision_summary = (
                counts["decision"]
                .fillna("Unknown")
                .value_counts()
            )

            st.subheader("Records by Decision")
            st.bar_chart(decision_summary)
    else:
        st.info(
            "No defect statistics are available yet."
        )


# =========================================================
# 5. REPORTS
# =========================================================
elif page == "Reports":
    st.title("Inspection Reports")

    st.write(
        "View and download saved inspection records."
    )

    history = get_history()

    if isinstance(history, pd.DataFrame) and not history.empty:
        st.metric("Available Inspection Records", len(history))

        st.subheader("Saved Inspection History")
        show_table(history)

        st.download_button(
            "Download Inspection History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="inspection_history.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No saved reports are available yet. "
            "Run an inspection first."
        )


# =========================================================
# 6. MODEL HEALTH
# =========================================================
elif page == "Model Health":
    st.title("Model Health")
    st.caption("Environment and model-file checks.")

    col1, col2, col3 = st.columns(3)

    col1.metric("OpenCV", cv2.__version__)
    col2.metric("Python", platform.python_version())

    try:
        import ultralytics

        col3.metric("Ultralytics", ultralytics.__version__)
        st.success("Ultralytics is installed.")
    except ImportError:
        col3.metric("Ultralytics", "Not installed")
        st.warning(
            "Ultralytics is not installed in this environment."
        )

    model_candidates = [
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "best.pt",
    ]

    found_model = next(
        (path for path in model_candidates if path.exists()),
        None,
    )

    if found_model:
        st.success(
            f"Model file found: {found_model.relative_to(BASE_DIR)}"
        )
    else:
        st.warning(
            "No model file was found in the standard locations. "
            "Confirm the trained model location with Member A."
        )

    st.subheader("Pipeline Check")

    try:
        get_pipeline()
        st.success("Pipeline module imported successfully.")
    except Exception as error:
        st.error("Could not import the inspection pipeline.")
        st.exception(error)

    st.subheader("Storage Check")

    try:
        get_storage()
        st.success("Storage functions imported successfully.")
    except Exception as error:
        st.error("Could not import storage functions.")
        st.exception(error)
