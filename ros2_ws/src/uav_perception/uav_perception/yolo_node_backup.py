
import json
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_detector")

        self.declare_parameter("model", "yolo11n.pt")
        self.declare_parameter(
            "output_video",
            "/home/anil/uav_object_detection/annotated_sitl_video.mp4",
        )
        self.declare_parameter(
            "output_log",
            "/home/anil/uav_object_detection/detection_telemetry.jsonl",
        )
        self.declare_parameter("output_fps", 10.0)

        model_path = self.get_parameter("model").value
        self.output_video = self.get_parameter("output_video").value
        self.output_log = self.get_parameter("output_log").value
        self.output_fps = float(self.get_parameter("output_fps").value)

        Path(self.output_video).parent.mkdir(
            parents=True, exist_ok=True
        )
        Path(self.output_log).parent.mkdir(
            parents=True, exist_ok=True
        )

        self.model = YOLO(model_path, task="detect")
        self.bridge = CvBridge()
        self.latest_telemetry = {}
        self.video_writer = None
        self.writer_failed = False
        self.log_file = open(
            self.output_log, "w", encoding="utf-8"
        )

        self.telemetry_sub = self.create_subscription(
            String,
            "/uav/telemetry",
            self.telemetry_callback,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            1,
        )

        self.image_pub = self.create_publisher(
            Image, "/detections/image", 10
        )
        self.detection_pub = self.create_publisher(
            String, "/detections", 10
        )

        self.get_logger().info("YOLO + SITL telemetry node started")
        self.get_logger().info(f"Video output: {self.output_video}")
        self.get_logger().info(f"Detection log: {self.output_log}")

    def telemetry_callback(self, msg):
        try:
            self.latest_telemetry = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid telemetry JSON")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )

            result = self.model.predict(
                frame,
                imgsz=320,
                conf=0.35,
                verbose=False,
            )[0]

            # Draw YOLO bounding boxes, class names and confidence.
            annotated = result.plot()

            telemetry = dict(self.latest_telemetry)

            # Overlay simulated flight telemetry on the image.
            if telemetry.get("connected"):
                lines = [
                    f"ALT: {telemetry.get('relative_altitude_m')} m",
                    f"HEADING: {telemetry.get('heading_deg')} deg",
                    f"SPEED: {telemetry.get('ground_speed_m_s')} m/s",
                    f"MODE: {telemetry.get('flight_mode', 'UNKNOWN')}",
                ]
            else:
                lines = ["SITL TELEMETRY: WAITING"]

            y = 30
            for line in lines:
                cv2.putText(
                    annotated,
                    line,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 30

            # Initialize MP4 writer using the first frame's dimensions.
            if self.video_writer is None and not self.writer_failed:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self.output_video,
                    fourcc,
                    self.output_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    writer.release()
                    self.writer_failed = True
                    self.get_logger().error(
                        "Could not open video output for writing"
                    )
                else:
                    self.video_writer = writer

            if self.video_writer is not None:
                self.video_writer.write(annotated)

            # Preserve the input frame's timestamp.
            stamp = {
                "sec": msg.header.stamp.sec,
                "nanosec": msg.header.stamp.nanosec,
            }

            detections = []
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                detections.append({
                    "class": result.names[class_id],
                    "confidence": round(
                        float(box.conf[0].item()), 3
                    ),
                    "xyxy": [
                        round(float(v), 1)
                        for v in box.xyxy[0].tolist()
                    ],
                })

            record = {
                "frame_stamp": stamp,
                "detections": detections,
                "telemetry": telemetry,
            }

            # One JSON object per processed frame.
            self.log_file.write(json.dumps(record) + "\n")
            self.log_file.flush()

            # Publish annotated image to ROS 2.
            output_msg = self.bridge.cv2_to_imgmsg(
                annotated, encoding="bgr8"
            )
            output_msg.header = msg.header
            self.image_pub.publish(output_msg)

            # Publish detections plus their associated telemetry.
            summary = String()
            summary.data = json.dumps(record)
            self.detection_pub.publish(summary)

        except Exception as exc:
            self.get_logger().error(
                f"Frame processing failed: {exc}"
            )

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if not self.log_file.closed:
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

