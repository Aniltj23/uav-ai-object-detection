
import cv2
from ultralytics import YOLO # Change these paths to your actual files
VIDEO_PATH = "/home/anil/uav_object_detection/aerial3.mp4"
MODEL_PATH = (
    "/home/anil/uav_object_detection/"
    "models/best.onnx"
)
OUTPUT_PATH = (
    "/home/anil/uav_object_detection/"
    "annotated_output.mp4"
)

# Load your fine-tuned ONNX model
model = YOLO(MODEL_PATH, task="detect")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Use a fallback if the video does not report a valid FPS
if fps <= 0:
    fps = 25.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_PATH, fourcc, fps, (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

frame_number = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run detection on this frame
        result = model.predict(
            frame,
            imgsz=320,
            conf=0.35,
            verbose=False
        )[0]

        # Draw boxes, labels and confidence scores
        annotated_frame = result.plot()

        # Save the annotated frame
        writer.write(annotated_frame)
        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Processed {frame_number}/{total_frames} frames"
            )

finally:
    cap.release()
    writer.release()

print(f"Finished! Saved video to: {OUTPUT_PATH}")

