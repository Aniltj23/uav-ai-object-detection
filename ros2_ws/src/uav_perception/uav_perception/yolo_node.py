import json
from collections import deque
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

        self.declare_parameter(
            "output_video",
            "/home/anil/uav_object_detection/annotated_sitl_video.mp4",
        )
        self.declare_parameter(
            "output_log",
            "/home/anil/uav_object_detection/detection_telemetry.jsonl",
        )
        self.declare_parameter(
            "model", "/home/anil/uav_object_detection/models/best.onnx"
        )
        self.declare_parameter("output_fps", 30.00)
        self.declare_parameter("tracker", "/home/anil/uav_object_detection/config/botsort_reid.yaml")
        self.declare_parameter("telemetry_sync_tolerance_ms", 500.0)
        self.declare_parameter("telemetry_buffer_size", 200)

        model_path = str(self.get_parameter("model").value)
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.output_video = str(self.get_parameter("output_video").value)
        self.output_log = str(self.get_parameter("output_log").value)
        self.output_fps = float(self.get_parameter("output_fps").value)
        self.tracker = str(self.get_parameter("tracker").value)
        self.sync_tolerance_ns = int(
            float(self.get_parameter("telemetry_sync_tolerance_ms").value) * 1_000_000
        )
        buffer_size = max(1, int(self.get_parameter("telemetry_buffer_size").value))

        if self.output_fps <= 0:
            raise ValueError("output_fps must be greater than zero")

        Path(self.output_video).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_log).parent.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(model_path, task="detect")
        self.bridge = CvBridge()
        # Each item stores the timestamp in ROS nanoseconds and a telemetry copy.
        self.telemetry_buffer = deque(maxlen=buffer_size)
        self.video_writer = None
        self.writer_failed = False
        self.tracking_id_warning_logged = False
        self.log_file = open(self.output_log, "w", encoding="utf-8")

        self.telemetry_sub = self.create_subscription(
            String, "/uav/telemetry", self.telemetry_callback, 50
        )
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 1
        )
        self.image_pub = self.create_publisher(Image, "/detections/image", 10)
        self.detection_pub = self.create_publisher(String, "/detections", 10)
        self.frame_ack_pub = self.create_publisher(
             String, "/camera/frame_processed", 10
        )

        self.get_logger().info("YOLO + persistent object tracking + telemetry sync started")
        self.get_logger().info(f"Video output: {self.output_video} ({self.output_fps:g} FPS)")
        self.get_logger().info(f"Detection log: {self.output_log}")
        self.get_logger().info(f"Loaded YOLO model: {Path(model_path).resolve()}")
        self.get_logger().info(f"Tracker: {self.tracker}")
        self.get_logger().info(
            f"Telemetry matching tolerance: {self.sync_tolerance_ns / 1_000_000:g} ms"
        )

    @staticmethod
    def stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def telemetry_stamp_ns(data, receipt_ns):
        """Use an explicit ROS-style timestamp if present; otherwise receipt time."""
        stamp = data.get("stamp")
        if isinstance(stamp, dict) and "sec" in stamp:
            try:
                return int(stamp["sec"]) * 1_000_000_000 + int(stamp.get("nanosec", 0))
            except (TypeError, ValueError):
                pass
        if "timestamp_sec" in data:
            try:
                return int(data["timestamp_sec"]) * 1_000_000_000 + int(
                    data.get("timestamp_nanosec", 0)
                )
            except (TypeError, ValueError):
                pass
        return receipt_ns

    def telemetry_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("telemetry JSON must be an object")
            receipt_ns = self.stamp_to_ns(self.get_clock().now().to_msg())
            sample_ns = self.telemetry_stamp_ns(data, receipt_ns)
            self.telemetry_buffer.append(
                {"stamp_ns": sample_ns, "receipt_ns": receipt_ns, "data": data}
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"Invalid telemetry JSON: {exc}")

    def match_telemetry(self, frame_stamp_ns):
        if not self.telemetry_buffer:
            return {}, None, None

        # Match the nearest sample in time, not merely the last message received.
        sample = min(
            self.telemetry_buffer,
            key=lambda item: abs(item["stamp_ns"] - frame_stamp_ns),
        )
        delta_ns = sample["stamp_ns"] - frame_stamp_ns
        if abs(delta_ns) > self.sync_tolerance_ns:
            return {}, delta_ns / 1_000_000, False

        return dict(sample["data"]), delta_ns / 1_000_000, True

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_stamp_ns = self.stamp_to_ns(msg.header.stamp)

            # persist=True keeps tracker state across consecutive frames.
            result = self.model.track(
                frame,
                imgsz=512,
                conf=0.25,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )[0]
            annotated = result.plot()

            telemetry, telemetry_delta_ms, telemetry_synced = self.match_telemetry(
                frame_stamp_ns
            )

            if telemetry_synced and telemetry.get("connected"):
                lines = [
                    f"ALT: {telemetry.get('relative_altitude_m')} m",
                    f"HEADING: {telemetry.get('heading_deg')} deg",
                    f"SPEED: {telemetry.get('ground_speed_m_s')} m/s",
                    f"MODE: {telemetry.get('flight_mode', 'UNKNOWN')}",
                ]
            elif telemetry_synced:
                lines = ["SITL TELEMETRY: DISCONNECTED"]
            else:
                lines = ["SITL TELEMETRY: NOT TIME-SYNCED"]

            y = 30
            for line in lines:
                cv2.putText(
                    annotated, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA,
                )
                y += 30

            if self.video_writer is None and not self.writer_failed:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self.output_video, fourcc, self.output_fps, (width, height)
                )
                if not writer.isOpened():
                    writer.release()
                    self.writer_failed = True
                    self.get_logger().error("Could not open video output for writing")
                else:
                    self.video_writer = writer

            if self.video_writer is not None:
                self.video_writer.write(annotated)

            detections = []
            boxes = result.boxes
            ids = boxes.id if boxes is not None else None
            for index, box in enumerate(boxes if boxes is not None else []):
                class_id = int(box.cls[0].item())
                track_id = None
                if ids is not None and index < len(ids):
                    track_id = int(ids[index].item())
                elif not self.tracking_id_warning_logged:
                    self.get_logger().warning(
                        "Tracker returned detections without IDs. Check tracker/model compatibility."
                    )
                    self.tracking_id_warning_logged = True

                detections.append({
                    "class": result.names[class_id],
                    "confidence": round(float(box.conf[0].item()), 3),
                    "track_id": track_id,
                    "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
                })

            processing_stamp = self.get_clock().now().to_msg()
            record = {
                "frame_stamp": {
                    "sec": msg.header.stamp.sec,
                    "nanosec": msg.header.stamp.nanosec,
                },
                "processing_stamp": {
                    "sec": processing_stamp.sec,
                    "nanosec": processing_stamp.nanosec,
                },
                "frame_id": msg.header.frame_id,
                "telemetry_sync": {
                    "matched": telemetry_synced,
                    "offset_ms": round(telemetry_delta_ms, 3)
                    if telemetry_delta_ms is not None else None,
                    "tolerance_ms": self.sync_tolerance_ns / 1_000_000,
                },
                "detections": detections,
                "telemetry": telemetry,
            }

            self.log_file.write(json.dumps(record) + "\n")
            self.log_file.flush()

            output_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            output_msg.header = msg.header

            self.image_pub.publish(output_msg)

            summary = String()
            summary.data = json.dumps(record)
            self.detection_pub.publish(summary)

        except Exception as exc:
            self.get_logger().error(f"Frame processing failed: {exc}")
        finally:
             ack = String()
             ack.data = json.dumps({
                  "sec": int(msg.header.stamp.sec),
                  "nanosec": int(msg.header.stamp.nanosec),
             })
             self.frame_ack_pub.publish(ack)

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
