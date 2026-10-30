import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pymavlink import mavutil


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("uav_telemetry")

        self.declare_parameter("connection_string", "udpin:127.0.0.1:14552")
        self.declare_parameter("publish_rate_hz", 10.0)

        connection_string = str(self.get_parameter("connection_string").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be greater than zero")

        self.connection = mavutil.mavlink_connection(connection_string)
        self.get_logger().info("Waiting for ArduPilot heartbeat...")
        heartbeat = self.connection.wait_heartbeat(timeout=15)
        if heartbeat is None:
            self.connection.close()
            raise RuntimeError(
                "No MAVLink heartbeat received. Check SITL and MAVProxy UDP forwarding."
            )

        self.get_logger().info("Connected to ArduPilot SITL")
        self.state = {
            "connected": True,
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "relative_altitude_m": None,
            "heading_deg": None,
            "ground_speed_m_s": None,
            "flight_mode": "UNKNOWN",
            # Source-clock value from GLOBAL_POSITION_INT, useful for diagnostics.
            # Do not compare this directly to ROS timestamps: it is time since boot.
            "source_time_boot_ms": None,
        }
        self.sequence = 0
        self.last_packet_receipt_stamp = None

        self.publisher = self.create_publisher(String, "/uav/telemetry", 10)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.update_telemetry)
        self.get_logger().info(
            f"Publishing timestamped telemetry on /uav/telemetry at {publish_rate_hz:g} Hz"
        )

    def update_telemetry(self):
        # Drain all currently available MAVLink messages before publishing one snapshot.
        while True:
            msg = self.connection.recv_match(blocking=False)
            if msg is None:
                break

            # Timestamp packet reception using the same ROS clock domain used by video_node.
            received_at = self.get_clock().now().to_msg()
            self.last_packet_receipt_stamp = {
                "sec": int(received_at.sec),
                "nanosec": int(received_at.nanosec),
            }

            msg_type = msg.get_type()
            if msg_type == "GLOBAL_POSITION_INT":
                self.state["latitude"] = msg.lat / 1e7
                self.state["longitude"] = msg.lon / 1e7
                self.state["altitude_m"] = msg.alt / 1000.0
                self.state["relative_altitude_m"] = msg.relative_alt / 1000.0
                if msg.hdg != 65535:
                    self.state["heading_deg"] = msg.hdg / 100.0

                # MAVLink GLOBAL_POSITION_INT velocity components are in cm/s.
                vx = msg.vx / 100.0
                vy = msg.vy / 100.0
                self.state["ground_speed_m_s"] = math.hypot(vx, vy)
                self.state["source_time_boot_ms"] = int(msg.time_boot_ms)

            elif msg_type == "VFR_HUD":
                self.state["ground_speed_m_s"] = float(msg.groundspeed)
                self.state["heading_deg"] = float(msg.heading)

            elif msg_type == "HEARTBEAT":
                # Ignore GCS heartbeats if present; update mode from autopilot heartbeat.
                if getattr(msg, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID) != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                    self.state["flight_mode"] = mavutil.mode_string_v10(msg)

        # This timestamp represents the telemetry snapshot's ROS publication time.
        # It is comparable to video_node's ROS frame timestamps, but is not the
        # original sensor measurement time. source_time_boot_ms is retained separately.
        stamp = self.get_clock().now().to_msg()
        self.sequence += 1
        output_data = dict(self.state)
        output_data.update({
            "stamp": {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)},
            "timestamp_sec": int(stamp.sec),
            "timestamp_nanosec": int(stamp.nanosec),
            "timestamp_source": "ros_snapshot_publish_time",
            "sequence": self.sequence,
            "last_packet_receipt_stamp": self.last_packet_receipt_stamp,
        })

        output = String()
        output.data = json.dumps(output_data, allow_nan=False)
        self.publisher.publish(output)

    def destroy_node(self):
        if hasattr(self, "connection") and self.connection is not None:
            self.connection.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TelemetryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
