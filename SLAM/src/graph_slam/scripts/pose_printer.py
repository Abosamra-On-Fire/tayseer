#!/usr/bin/env python3
"""
pose_printer.py
Subscribes to /amcl_pose and logs x, y, yaw every second.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class PosePrinter(Node):
    def __init__(self):
        super().__init__('pose_printer')
        # NOTE: do NOT declare 'use_sim_time' — ROS 2 Humble declares it
        # automatically. Declaring it again raises ParameterAlreadyDeclaredException.

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_cb,
            10,
        )
        self.create_timer(1.0, self.timer_cb)
        self.latest = None
        self.get_logger().info('pose_printer started — waiting for /amcl_pose …')

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        self.latest = msg

    def timer_cb(self):
        if self.latest is None:
            self.get_logger().info('waiting for /amcl_pose …')
            return

        x = self.latest.pose.pose.position.x
        y = self.latest.pose.pose.position.y
        q = self.latest.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.get_logger().info(
            f'x={x:.3f}  y={y:.3f}  yaw={math.degrees(yaw):.1f}°'
        )


def main(args=None):
    rclpy.init(args=args)
    node = PosePrinter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()