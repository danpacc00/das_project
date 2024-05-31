import sys

import numpy as np
import phi
import rclpy
from costs_fn import SurveillanceCost
from geometry_msgs.msg import Pose
from message.msg import NodeData
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
        self.target_position = np.array(self.get_parameter("target").value)
        self.initial_pose = np.array(self.get_parameter("initial_pose").value)
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
                NodeData, f"/warden_{neighbor}", self._neighbor_callback, 10
            )

        self._publisher = self.create_publisher(NodeData, f"/warden_{self.id}", 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self._simtime = 0
        self._data = []
        self._phi_fn = phi.Identity()
        self._cost_fn = SurveillanceCost(tradeoff=1.0)
        self._zz = [self.initial_pose]
        self._ss = [{jj: np.zeros(2) for jj in self.neighbors}]
        self._vv = [{jj: np.zeros(2) for jj in self.neighbors}]

        self._ss[self._simtime][self.id] = self.phi_fn(self._zz[self._simtime])[0]
        self._vv[self._simtime][self.id] = self.cost_fn(
            self.target_position, self._zz[self._simtime][self.id], self._ss[self._simtime][self.id]
        )[2]

    def _neighbor_callback(self, msg):
        neighbor_id = msg.warden_id
        ss = msg.ss
        vv = msg.vv

        if self._data[self._simtime] is None:
            self._data.append({})

        self._ss[self._simtime][neighbor_id] = ss
        self._vv[self._simtime][neighbor_id] = vv
        self._info(f"Received data from neighbor {neighbor_id}")

    def _timer_callback(self):
        msg = NodeData()
        msg.warden_id = self.id

        if self._simtime == 0:
            msg.ss = self._ss[self._simtime][self.id]
            msg.vv = self._vv[self._simtime][self.id]
            self._publisher.publish(msg)
            self._info(f"Published initial pose (x = {self.pose.position.x}, y = {self.pose.position.y})")
            self._simtime += 1
            return

        all_received = all(neighbor_id in self._data[self._simtime] for neighbor_id in self.neighbors)
        if not all_received:
            self._info("Waiting for all neighbors to respond...")
            return

        self._info("All neighbors have responded. Updating pose...")
        self._update()
        msg.ss = self._ss[self._simtime][self.id]
        msg.vv = self._vv[self._simtime][self.id]
        self._publisher.publish(msg)
        self._info("Published updated data")

        self._simtime += 1

    def _update(self):
        self._info("Updating...")

        li_nabla_1 = self._cost_fn(
            self.target_position, self._zz[self._simtime - 1], self._ss[self._simtime - 1][self.id]
        )[1]
        _, phi_grad = self._phi_fn(self._zz[self._simtime - 1])

        self._zz[self._simtime] = self._zz[self._simtime] - self.alpha * (
            li_nabla_1 + self._vv[self._simtime][self.id] + phi_grad
        )

        self._ss[self._simtime][self.id] += (
            self.phi_fn(self._zz[self._simtime])[0] - self.phi_fn(self._zz[self._simtime - 1])[0]
        )
        self._vv[self._simtime][self.id] += (
            self.phi_fn(self._zz[self._simtime])[1] - self.phi_fn(self._zz[self._simtime - 1])[1]
        )

        for jj, weight in self.neighbors.items():
            self._ss[self._simtime][self.id] += weight * self._ss[self._simtime - 1][jj]
            self._vv[self._simtime][self.id] += weight * self._vv[self._simtime - 1][jj]

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
