# UAV AI Object Detection and Tracking System

A simulation-based UAV perception project integrating **ROS 2 Jazzy,
ArduPilot SITL, YOLO, BoT-SORT, and MAVLink telemetry** to detect
and track objects in aerial video while associating detections with
simulated UAV telemetry.

The project uses prerecorded aerial video for visual perception and
ArduPilot Software-in-the-Loop (SITL) for simulated flight telemetry. It
does not require a physical UAV for this simulation.

------------------------------------------------------------------------

## Table of Contents

-   [Project Overview](#project-overview)
-   [Key Features](#key-features)
-   [System Architecture](#system-architecture)
-   [Technology Stack](#technology-stack)
-   [Repository Structure](#repository-structure)
-   [Prerequisites](#prerequisites)
-   [Installation and Configuration](#installation-and-configuration)
-   [Running the System](#running-the-system)
-   [ROS 2 Topics](#ros-2-topics)
-   [Outputs](#outputs)
-   [Demonstration](#demonstration)
-   [Development Status](#development-status)
-   [Future Work](#future-work)
-   [Project Scope](#project-scope)
-   [Acknowledgements](#acknowledgements)
-   [Author](#author)

------------------------------------------------------------------------

## Project Overview

Detecting objects in aerial imagery is challenging because objects may
occupy only a small number of pixels, appear at different scales, and
move between video frames. This project combines object detection, multi-object tracking, and simulated UAV telemetry in
a modular ROS 2 pipeline.

A custom-trained YOLO model exported to ONNX format performs object
detection. BoT-SORT
associates detections across frames and maintains tracking identities.
Separately, a ROS 2 telemetry node receives MAVLink data from ArduPilot
SITL. Detection and telemetry timestamps are matched within a
configurable tolerance.

## Key Features

-   **Custom YOLO model:** Uses a YOLO11s model fine-tuned on VisDrone dataset exported in ONNX
    format.
-   **Multi-object tracking:** Integrates BoT-SORT to associate
    detections across video frames.
-   **Appearance-based ReID:** Supports appearance-based
    association using a configured ReID model
-   **ROS 2 architecture:** Separates video input, perception, and
    telemetry into ROS 2 nodes.
-   **ArduPilot SITL integration:** Receives simulated UAV telemetry
    through MAVLink.
-   **Timestamp-based synchronization:** Matches detections with
    temporally close telemetry samples.
-   **Annotated output:** Produces annotated video and structured
    detection/telemetry logs.

------------------------------------------------------------------------

## System Architecture

``` text
                 PRERECORDED AERIAL VIDEO
                            |
                            v
                    ROS 2 VIDEO NODE
                            |
                            v
                     IMAGE FRAMES
                            |
                            v
                       YOLO (ONNX)
                            |
                            v
                    OBJECT DETECTIONS
                            |
                            v
                    BoT-SORT TRACKING
                            |
                            v
                  TRACKED OBJECT RESULTS
                            |
                            v
                 TIMESTAMP-BASED MATCHING
                            ^
                            |
                    ROS 2 TELEMETRY NODE
                            ^
                            |
                         MAVLink
                            ^
                            |
                     ARDUPILOT SITL

                            |
                            v
                 ANNOTATED VIDEO AND LOGS
```

### Processing Workflow

1.  A prerecorded aerial video is read by the ROS 2 video node.
2.  Video frames are published to the perception pipeline.
3.  The custom YOLO ONNX model performs object detection on the slices.
4.  Detection results are merged and passed to BoT-SORT for multi-object
    tracking.
5.  ArduPilot SITL provides simulated UAV telemetry over MAVLink.
6.  The ROS 2 telemetry node receives and publishes telemetry
    information.
7.  Detection timestamps are matched with nearby telemetry timestamps
    using a configurable synchronization tolerance.
8.  The system generates annotated video and structured
    detection/telemetry records.

------------------------------------------------------------------------

## Technology Stack

  Area                      Technology
  ------------------------- ----------------------
  Programming language      Python
  Robotics middleware       ROS 2 Jazzy
  UAV simulation            ArduPilot SITL
  UAV communication         MAVLink
  Telemetry library         pymavlink
  Object detection          Custom YOLO model
  Model format              ONNX
  Multi-object tracking     BoT-SORT
  Computer vision           OpenCV
  ROS image conversion      CvBridge
  Development environment   Ubuntu 24.04 on WSL2

------------------------------------------------------------------------

## Repository Structure

``` text
uav-ai-object-detection/
│
├── README.md
├── .gitignore
│
├── config/
│   ├── botsort_reid.yaml
│   └── bytetrack_custom.yaml
│
├── dashboard/
│   └── app.py
│
├── ros2_ws/
│   └── src/
│       └── uav_perception/
│           ├── package.xml
│           ├── setup.py
│           ├── setup.cfg
│           ├── resource/
│           │   └── uav_perception
│           ├── test/
│           └── uav_perception/
│               ├── __init__.py
│               ├── video_node.py
│               ├── telemetry_node.py
│               ├── yolo_node.py
│               └── yolo_node_backup.py
│
├── save_annotated_video.py
└── test_detection.py
```

------------------------------------------------------------------------

## Prerequisites

The development environment uses:

-   Windows 11 with WSL2
-   Ubuntu 24.04
-   ROS 2 Jazzy
-   ArduPilot SITL
-   MAVProxy
-   Python and the project dependencies
-   A trained YOLO model exported to ONNX format

ArduPilot is maintained separately from this repository. Follow the
official ArduPilot documentation to install its source code,
dependencies, and SITL tools.

------------------------------------------------------------------------

## Installation and Configuration

### 1. Clone this repository

``` bash
git clone https://github.com/Aniltj23/uav-ai-object-detection.git
cd uav-ai-object-detection
```

### 2. Install ROS 2 Jazzy

Install ROS 2 Jazzy for Ubuntu 24.04 using the official instructions:

https://docs.ros.org/en/jazzy/Installation.html

Source the ROS 2 environment:

``` bash
source /opt/ros/jazzy/setup.bash
```

### 3. Build the ROS 2 package

From the repository root:

``` bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

If the project uses a Python virtual environment, activate it in the
terminal before running the relevant Python nodes. Ensure that the
Python packages used by ROS 2, especially OpenCV and CvBridge, are
compatible with the installed ROS 2 environment.

### 4. Install and configure ArduPilot SITL

ArduPilot is a separate dependency and is not included in this
repository.

Official repository:

https://github.com/ArduPilot/ardupilot

Follow the official ArduPilot SITL setup instructions to clone the
source, install prerequisites, build the desired vehicle, and start the
simulator.

The telemetry node in this project is configured to connect to:

``` text
udpin:127.0.0.1:14552
```

Configure SITL/MAVProxy to forward MAVLink telemetry to the same UDP
endpoint. The simulator and telemetry node must be able to communicate
over that endpoint.

### 5. Add the trained YOLO model

The trained model is not included in this repository.

Place your ONNX model in the expected local location:

``` text
models/
└── best.onnx
```

The current development configuration uses:

``` text
/home/anil/uav_object_detection/models/best.onnx
```

If you use a different location, update the model path in the YOLO node
or its ROS 2 parameters accordingly.

### 6. Configure the tracker

Tracker configuration files are stored in `config/`.

The project includes BoT-SORT and ByteTrack configuration files. Select
the intended tracker in the YOLO node and verify that the configured
tracker file and any required ReID model are available.

------------------------------------------------------------------------

## Running the System

The complete workflow requires ArduPilot SITL, the ROS 2
video/perception nodes and the telemetry node

Before starting:

1.  Activate the appropriate Python environment.
2.  Source `/opt/ros/jazzy/setup.bash`.
3.  Build and source the workspace if it has not already been built.
4.  Confirm that the input video and ONNX model exist at the configured
    paths.
5.  Start ArduPilot SITL and configure MAVLink forwarding to
    `127.0.0.1:14552`.
6.  Start the ROS 2 video, perception, and telemetry nodes using the
    executable names and parameters defined by the package.


**Important:** Exact ROS 2 executable names, launch commands, and
parameters depend on the current implementation. Use
the package configuration as the source of truth
rather than assuming generic launch commands.

------------------------------------------------------------------------

## ROS 2 Topics

The current implementation uses the following topic names:

  Topic                       Purpose
  --------------------------- ------------------------------
  `/camera/image_raw`         Input video frames
  `/uav/telemetry`            UAV telemetry
  `/detections/image`         Annotated image output
  `/detections`               Object detection results
  `/camera/frame_processed`   Processed-frame notification

Verify topic types and message definitions in the package before
integrating additional ROS 2 nodes.

------------------------------------------------------------------------

## Outputs

### Annotated video

The annotated video can include:

-   Detected object bounding boxes
-   Object class labels
-   Detection confidence values, where displayed
-   Tracking IDs
-   UAV telemetry overlays when synchronized telemetry is available

### Detection and telemetry log

Detection and telemetry records are saved in JSON Lines format.
Depending on the implementation, records can contain information such
as:

-   Frame timestamp
-   Object class
-   Detection confidence
-   Bounding box coordinates
-   Tracking ID
-   UAV altitude
-   Heading
-   Speed
-   Flight mode

### Object detection

<img width="1366" height="768" alt="Detection 1" src="https://github.com/user-attachments/assets/2a6aff52-cee6-4bfd-8ead-8a17bef70bdc" />


### Multi-object tracking

<img width="1366" height="768" alt="Detection 2" src="https://github.com/user-attachments/assets/dd00c360-1ee5-4e4e-8b1e-af9d10a3f61d" />

------------------------------------------------------------------------

## Development Status

  Component                                Status
  ---------------------------------------- -------------------
  Custom YOLO model integration            Implemented
  ONNX inference                           Implemented
  BoT-SORT tracking                        Integrated
  ROS 2 perception package                 Implemented
  ArduPilot SITL connection                Integrated
  MAVLink telemetry collection             Integrated
  Detection/telemetry timestamp matching   Implemented
  Annotated video generation               Implemented

------------------------------------------------------------------------

## Future Work

-   Improve inference throughput and processing efficiency.
-   Evaluate full-frame YOLO against SAHI using detection metrics and
    processing time.
-   Improve tracking consistency for small and partially occluded
    objects.
-   Evaluate tracking performance using standard multi-object tracking
    metrics.
-   Improve telemetry alignment and handling of missing telemetry
    samples.
-   Add reproducible installation and launch scripts.
-   Explore natural-language-driven object search and UAV mission
    instructions.

------------------------------------------------------------------------

## Project Scope and Limitations

This is a **simulation-based UAV AI perception project**.

-   Visual perception is performed on prerecorded aerial video.
-   UAV telemetry is obtained from ArduPilot SITL.
-   The project does not currently demonstrate autonomous control of a
    physical UAV.
-   Processing speed depends on the available hardware and inference
    configuration.
-   Small-object detection and tracking performance depend on video
    quality, model training, slice settings, and tracker configuration.

------------------------------------------------------------------------

## Acknowledgements

This project uses and builds upon the following open-source
technologies:

-   [Ultralytics](https://github.com/ultralytics/ultralytics)
-   [SAHI](https://github.com/obss/sahi)
-   [ArduPilot](https://github.com/ArduPilot/ardupilot)
-   [ROS 2](https://github.com/ros2)
-   [OpenCV](https://github.com/opencv/opencv)

------------------------------------------------------------------------

## Author

GitHub: [@Aniltj23](https://github.com/Aniltj23)

Project repository:
[uav-ai-object-detection](https://github.com/Aniltj23/uav-ai-object-detection)
