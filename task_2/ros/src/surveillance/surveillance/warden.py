import sys

import numpy as np
import phi
import rclpy
from aggregative_tracking import NodeUpdater
from costs_fn import SurveillanceCost
from geometry_msgs.msg import Pose
from message.msg import Position
from rclpy.node import Node

DEFAULT_TIMER_PERIOD = 2


class Warden(Node):
    def __init__(self):
        super().__init__(
            "warden", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True
        )

        self.id = self.get_parameter("id").value
        self.weights = self.get_parameter("weights").value
        neighbors = self.get_parameter("neighbors").value
        self.neighbors = {neighbor: weight for neighbor, weight in zip(neighbors, self.weights)}
        self.target_position = self.get_parameter("target").value
        self.initial_pose = self.get_parameter("initial_pose").value
        self.timer_period = self.get_parameter("timer_period").value or DEFAULT_TIMER_PERIOD

        self.pose = Pose()
        self.pose.position.x = self.initial_pose[0]
        self.pose.position.y = self.initial_pose[1]

        self.target = Pose()
        self.target.position.x = self.target_position[0]
        self.target.position.y = self.target_position[1]

        self._info(
            f"Starting warden {self.id} at position x = {self.pose.position.x}, y = {self.pose.position.y} with target x = {self.target.position.x}, y = {self.target.position.y}"
        )

        self._subscriptions = {}
        for neighbor in self.neighbors:
            self._info(f"Subscribing to neighbor {neighbor}")
            self._subscriptions[neighbor] = self.create_subscription(
                Position, f"/warden_{neighbor}", self._neighbor_callback, 10
            )

        self._publisher = self.create_publisher(Position, f"/warden_{self.id}", 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self._simtime = 0
        self._data = []
        self._cost_fn = SurveillanceCost(tradeoff=1.0)
        self._updater = NodeUpdater(self.id, self._cost_fn, phi.Identity(), self.neighbors, alpha=1e-2)

    def _neighbor_callback(self, msg):
        neighbor_id = msg.warden_id
        pose = msg.pose

        if self._data[self._simtime] is None:
            self._data.append({})

        self._data[self._simtime][neighbor_id] = pose
        self._info(f"Received pose from neighbor {neighbor_id} (x = {pose.position.x}, y = {pose.position.y})")

    def _timer_callback(self):
        msg = Position()
        msg.warden_id = self.id

        if self._simtime == 0:
            msg.pose = self.pose
            self._publisher.publish(msg)
            self._info(f"Published initial pose (x = {self.pose.position.x}, y = {self.pose.position.y})")
            self._simtime += 1
            return

        all_received = all(neighbor_id in self._data[self._simtime] for neighbor_id in self.neighbors)
        if not all_received:
            self._info("Waiting for all neighbors to respond...")
            return

        self._info("All neighbors have responded. Updating pose...")
        self._monitor()
        msg.pose = self.pose
        self._publisher.publish(msg)
        self._info(f"Published updated pose (x = {self.pose.position.x}, y = {self.pose.position.y})")

        self._simtime += 1

    def _monitor(self):
        self._info("Monitoring...")

        # TODO: add zz, ss, vv
        self.pose, _, _ = self._updater.update(self._simtime, self.target, self.pose, self._data[self._simtime])

    def _debug(self, msg):
        self.get_logger().debug(f"[{self.id}] {msg}")

    def _info(self, msg):
        self.get_logger().info(f"[{self.id}] {msg}")

    def _warn(self, msg):
        self.get_logger().warn(f"[{self.id}] {msg}")

    def _error(self, msg):
        self.get_logger().error(f"[{self.id}] {msg}")

    def _fatal(self, msg):
        self.get_logger().fatal(f"[{self.id}] {msg}")


def main(args=None):
    rclpy.init(args=args)

    try:
        warden = Warden()
        rclpy.spin(warden)
    except KeyboardInterrupt:
        pass
    except BaseException:
        print("Exception in warden:", file=sys.stderr)
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
