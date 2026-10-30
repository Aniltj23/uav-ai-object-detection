import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

# --------------------------------------------------
# Project configuration
# --------------------------------------------------

HOME = Path.home()

PROJECT = HOME / "uav_object_detection"
WORKSPACE = HOME / "uav_ros_ws"
ARDUPILOT = HOME / "ardupilot" / "ArduCopter"

VENV = PROJECT / ".venv"
MODEL = PROJECT / "models" / "best.onnx"
INPUT_VIDEO = PROJECT / "aerial3.mp4"

YOLO_NODE = (
    WORKSPACE / "src/uav_perception/uav_perception/yolo_node.py"
)
VIDEO_NODE = (
    WORKSPACE / "src/uav_perception/uav_perception/video_node.py"
)
TELEMETRY_NODE = (
    WORKSPACE / "src/uav_perception/uav_perception/telemetry_node.py"
)

OUTPUT_VIDEO = PROJECT / "annotated_sitl_video.mp4"
OUTPUT_LOG = PROJECT / "detection_telemetry.jsonl"

LOG_DIR = PROJECT / "dashboard" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="UAV AI Control Center",
    page_icon="🚁",
    layout="wide",
)

if "processes" not in st.session_state:
    st.session_state.processes = {}

if "run_started" not in st.session_state:
    st.session_state.run_started = None

if "last_error" not in st.session_state:
    st.session_state.last_error = ""


def is_running():
    return any(
        item["process"].poll() is None
        for item in st.session_state.processes.values()
    )


def launch_process(name, command, cwd):
    """Launch a process in its own process group and capture logs."""
    log_path = LOG_DIR / f"{name}.log"
    log_file = log_path.open("w", encoding="utf-8")

    try:
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    st.session_state.processes[name] = {
        "process": process,
        "log_file": log_file,
        "log_path": log_path,
    }


def stop_process(name):
    """Stop a managed process group, allowing graceful shutdown first."""
    item = st.session_state.processes.get(name)

    if not item:
        return

    process = item["process"]

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        except ProcessLookupError:
            pass

    item["log_file"].close()


def stop_pipeline():
    # Stop the video source first, then its consumers and SITL.
    for name in ("video", "yolo", "telemetry", "sitl"):
        stop_process(name)


def read_log(path, max_lines=50):
    if not path.exists():
        return "Log file not created yet."

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()

        return "\n".join(lines[-max_lines:]) or "Log is empty."
    except OSError as exc:
        return f"Unable to read log: {exc}"


def get_video_fps():
    capture = cv2.VideoCapture(str(INPUT_VIDEO))

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if fps <= 0:
        raise RuntimeError(
            f"Could not determine FPS of {INPUT_VIDEO}"
        )

    return round(fps, 3)


def start_pipeline():
    # Validate files before launching anything.
    required = [
        MODEL,
        INPUT_VIDEO,
        YOLO_NODE,
        VIDEO_NODE,
        TELEMETRY_NODE,
    ]

    missing = [str(path) for path in required if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    if not ARDUPILOT.is_dir():
        raise FileNotFoundError(f"ArduPilot directory missing: {ARDUPILOT}")

    if is_running():
        raise RuntimeError("A managed pipeline is already running.")

    # Close old log handles before replacing an earlier run.
    for item in st.session_state.processes.values():
        if not item["log_file"].closed:
            item["log_file"].close()

    st.session_state.processes = {}
    st.session_state.last_error = ""
    st.session_state.run_started = time.time()

    fps = get_video_fps()

    ros_setup = (
        "source /opt/ros/jazzy/setup.bash && "
        f"source {WORKSPACE}/install/setup.bash && "
        f"source {VENV}/bin/activate && "
    )

    try:
        
        SIM_VEHICLE = HOME / "ardupilot" / "Tools" / "autotest" / "sim_vehicle.py"
        # 1. Start ArduPilot SITL and forward telemetry to UDP 14552.

        sitl_command = (
            f"source {VENV}/bin/activate && "
            f"exec python3 {SIM_VEHICLE} "
            "-v ArduCopter "
            "--console "
            "--out=127.0.0.1:14552"
        )

        launch_process("sitl", sitl_command, ARDUPILOT)

        # Allow SITL/MAVProxy to initialize.
        time.sleep(10)

        if st.session_state.processes["sitl"]["process"].poll() is not None:
            raise RuntimeError(
                "SITL exited during startup. Check the SITL log."
            )

        # 2. Start the MAVLink telemetry subscriber.
        launch_process(
            "telemetry",
            ros_setup + f"exec python {TELEMETRY_NODE}",
            WORKSPACE,
        )

        time.sleep(2)

        # 3. Start YOLO using the trained ONNX model.
        yolo_command = (
            ros_setup
            + f"exec python {YOLO_NODE} "
            + "--ros-args "
            + f"-p model:={MODEL} "
            + f"-p output_fps:={fps}"
        )

        launch_process("yolo", yolo_command, WORKSPACE)

        time.sleep(2)

        if st.session_state.processes["yolo"]["process"].poll() is not None:
            raise RuntimeError(
                "YOLO exited during startup. Check the YOLO log."
            )

        # 4. Start the video publisher last, allowing subscribers to connect.
        launch_process(
            "video",
            ros_setup + f"exec python {VIDEO_NODE}",
            WORKSPACE,
        )

    except Exception:
        stop_pipeline()
        raise


# --------------------------------------------------
# Dashboard UI
# --------------------------------------------------

st.title("🚁 UAV AI Control Center")
st.caption("ROS 2 Jazzy · ArduPilot SITL · YOLO ONNX")

st.info(
    "Local simulation only. This dashboard manages the processes "
    "it starts; stop any manually launched copies before starting."
)

running = is_running()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pipeline", "RUNNING" if running else "STOPPED")

with col2:
    st.metric(
        "Model",
        "best.onnx" if MODEL.is_file() else "NOT FOUND",
    )

with col3:
    st.metric(
        "Input video",
        "READY" if INPUT_VIDEO.is_file() else "NOT FOUND",
    )

start_col, stop_col = st.columns(2)

with start_col:
    if st.button(
        "▶ Start pipeline",
        type="primary",
        disabled=running,
        use_container_width=True,
    ):
        try:
            with st.spinner("Starting SITL and ROS 2 nodes..."):
                start_pipeline()
            st.success("Startup commands completed.")
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.error(f"Pipeline startup failed: {exc}")

with stop_col:
    if st.button(
        "■ Stop pipeline",
        disabled=not st.session_state.processes,
        use_container_width=True,
    ):
        stop_pipeline()
        st.success("Stop requests sent to managed processes.")

if st.session_state.last_error:
    st.error(st.session_state.last_error)

st.subheader("Process status")

processes = st.session_state.processes

if not processes:
    st.caption("No processes have been started by this dashboard.")
else:
    for name, item in processes.items():
        code = item["process"].poll()

        if code is None:
            st.write(f"🟢 **{name.upper()}** — running")
        else:
            st.write(f"🔴 **{name.upper()}** — exited (code {code})")

st.subheader("Process logs")

for name in ("sitl", "telemetry", "yolo", "video"):
    log_path = LOG_DIR / f"{name}.log"

    with st.expander(f"{name.upper()} log", expanded=False):
        st.code(read_log(log_path), language="text")

st.subheader("Output files")

if is_running():
    st.info(
        "The pipeline is running. Stop it to finalize the MP4 before "
        "viewing the completed video."
    )
elif OUTPUT_VIDEO.exists():
    st.video(str(OUTPUT_VIDEO))

    st.download_button(
        "Download annotated video",
        data=OUTPUT_VIDEO.read_bytes(),
        file_name=OUTPUT_VIDEO.name,
        mime="video/mp4",
    )
else:
    st.caption("No annotated video has been generated yet.")

if OUTPUT_LOG.exists():
    st.download_button(
        "Download detection/telemetry JSONL",
        data=OUTPUT_LOG.read_bytes(),
        file_name=OUTPUT_LOG.name,
        mime="application/x-ndjson",
    )
