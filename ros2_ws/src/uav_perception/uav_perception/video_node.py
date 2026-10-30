
import json
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class VideoNode(Node):
    def __init__(self):
        super().__init__("drone_video_publisher")

        self.declare_parameter("video_path", "")
        path = self.get_parameter("video_path").value

        if not path:
            raise ValueError("Set the video_path parameter")

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps != fps:
            fps = 25.0

        self.fps = fps
        self.frame_index = 0
        self.pending_stamp = None
        self.video_ended = False

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Image, "/camera/image_raw", 10
        )

        self.ack_sub = self.create_subscription(
            String,
            "/camera/frame_processed",
            self.ack_callback,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.fps,
            self.publish_frame,
        )

        self.get_logger().info(
            f"Publishing {path} at {self.fps:.3f} FPS "
            "with frame-by-frame acknowledgement"
        )

    def ack_callback(self, msg):
        try:
            data = json.loads(msg.data)
            ack_stamp = (
                int(data["sec"]),
                int(data["nanosec"]),
            )

            if ack_stamp == self.pending_stamp:
                self.pending_stamp = None

        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.get_logger().warning("Received invalid frame acknowledgement")

    def publish_frame(self):
        if self.video_ended or self.pending_stamp is not None:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.video_ended = True
            self.get_logger().info(
                f"Video ended after {self.frame_index} frames"
            )
            self.cap.release()
            self.timer.cancel()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        stamp = self.get_clock().now().to_msg()
        msg.header.stamp = stamp
        msg.header.frame_id = "drone_camera"

        self.pending_stamp = (
            int(stamp.sec),
            int(stamp.nanosec),
        )

        self.publisher.publish(msg)
        self.frame_index += 1

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
