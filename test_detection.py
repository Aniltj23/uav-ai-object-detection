import cv2
from ultralytics import YOLO

MODEL_PATH = (
    "/home/anil/uav_object_detection/"
    "models/best.onnx"
)

VIDEO_PATH = (
    "/home/anil/uav_object_detection/"
    "aerial3.mp4"
)

model = YOLO(MODEL_PATH, task="detect")

cap = cv2.VideoCapture(VIDEO_PATH)

# Select one fixed frame.
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read the selected video frame.")

cv2.imwrite(
    "/home/anil/uav_object_detection/test_frame.jpg",
    frame,
)

# Repeat inference on exactly the same image.
# Test different resolution and confidence combinations
settings = [
    (320, 0.35),
    (320, 0.25),
    (320, 0.15),
    (640, 0.35),
    (640, 0.25),
    (640, 0.15),
]

for imgsz, conf in settings:
    result = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        verbose=False,
    )[0]

    print(
        f"imgsz={imgsz}, conf={conf}: "
        f"{len(result.boxes)} detections"
    )

    result.save(
        filename=(
            f"/home/anil/uav_object_detection/"
            f"test_img{imgsz}_conf{conf:.2f}.jpg"
        )
    )
