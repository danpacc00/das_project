import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from message.msg import NodeData, PlotterData
from rclpy.node import Node

from surveillance.costs_fn import SurveillanceCost
from surveillance.phi import Identity

DEFAULT_TIMER_PERIOD = 2  # seconds
DEFAULT_ALPHA = 1e-2  # stepsize


class Warden(Node):
    def __init__(self):
        super().__init__(
            "warden", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True
        )

        self.id = self.get_parameter("id").value
        self.weights = self.get_parameter("weights").value
        self.neighbors = self.get_parameter("neighbors").value
        self.target_position = np.array(self.get_parameter("target").value)
        self.initial_pose = np.array(self.get_parameter("initial_pose").value)
        self.timer_period = self.get_parameter("timer_period").value or DEFAULT_TIMER_PERIOD
        self.alpha = self.get_parameter("alpha").value or DEFAULT_ALPHA

        self.pose = Pose()
        self.pose.position.x = self.initial_pose[0]
        self.pose.position.y = self.initial_pose[1]

        self.target = Pose()
        self.target.position.x = self.target_position[0]
        self.target.position.y = self.target_position[1]

        self._debug(
            f"Starting warden {self.id} at position x = {self.pose.position.x}, y = {self.pose.position.y} with target x = {self.target.position.x}, y = {self.target.position.y}"
        )

        self._subs = {}
        for neighbor in self.neighbors:
            self._debug(f"Subscribing to neighbor {neighbor}")
            self._subs[neighbor] = self.create_subscription(
                NodeData, f"/warden_{neighbor}", self._neighbor_callback, 10
            )

        self._publisher = self.create_publisher(NodeData, f"/warden_{self.id}", 10)
        self._plotter_pub = self.create_publisher(PlotterData, "/plotter", 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self._simtime = 0
        self._data = []
        self._phi_fn = Identity()
        self._cost_fn = SurveillanceCost(tradeoff=1.0)
        self._zz = [self.initial_pose]
        self._ss = {jj: [np.zeros(2)] for jj in self.neighbors}
        self._vv = {jj: [np.zeros(2)] for jj in self.neighbors}

        self._ss[self.id] = [self._phi_fn(self._zz[self._simtime])[0]]
        self._vv[self.id] = [
            self._cost_fn(self.target_position, self._zz[self._simtime], self._ss[self.id][self._simtime])[2]
        ]

    def _neighbor_callback(self, msg):
        neighbor_id = msg.warden_id
        ss = msg.ss
        vv = msg.vv
        time = msg.time

        self._ss[neighbor_id].append(np.array(ss))
        self._vv[neighbor_id].append(np.array(vv))
        self._debug(f"Received data from neighbor {neighbor_id} for time {time}: ss = {ss}, vv = {vv}")

    def _timer_callback(self):
        msg = NodeData()
        msg.warden_id = self.id
        msg.time = self._simtime

        if self._simtime == 0:
            msg.ss = [self._ss[self.id][self._simtime][0], self._ss[self.id][self._simtime][1]]
            msg.vv = [self._vv[self.id][self._simtime][0], self._vv[self.id][self._simtime][1]]
            self._publisher.publish(msg)
            self._debug("Published initial data")

            time.sleep(1)

            data = PlotterData()
            data.warden_id = self.id
            data.zz = [float(self._zz[self._simtime][0]), float(self._zz[self._simtime][1])]
            data.cost = 0.0
            data.grad = 0.0
            self._plotter_pub.publish(data)
            self._info(f"Published data for plotter: zz = {self._zz[self._simtime]}")

            self._simtime += 1
            return

        all_received = all(len(self._ss[neighbor_id]) > self._simtime for neighbor_id in self.neighbors)
        if not all_received:
            self._debug("Waiting for all neighbors to respond...")
            return

        self._debug(f"All neighbors have responded. Updating pose for time {self._simtime}...")
        cost, grad = self._update()
        msg.ss = [self._ss[self.id][self._simtime][0], self._ss[self.id][self._simtime][1]]
        msg.vv = [self._vv[self.id][self._simtime][0], self._vv[self.id][self._simtime][1]]
        self._publisher.publish(msg)

        data = PlotterData()
        data.warden_id = self.id
        data.zz = [float(self._zz[self._simtime][0]), float(self._zz[self._simtime][1])]
        data.cost = cost
        data.grad = grad
        self._plotter_pub.publish(data)
        self._debug("Published data for plotter")

        self._simtime += 1

    def _update(self):
        self._debug("Updating...")
        kk = self._simtime
        ii = self.id

        li_nabla_1 = self._cost_fn(self.target_position, self._zz[kk - 1], self._ss[ii][kk - 1])[1]
        _, phi_grad = self._phi_fn(self._zz[kk - 1])

        self._zz.append(self._zz[kk - 1] - self.alpha * (li_nabla_1 + self._vv[ii][kk - 1] + phi_grad))

        self._ss[ii].append(self._phi_fn(self._zz[kk])[0] - self._phi_fn(self._zz[kk - 1])[0])
        self._vv[ii].append(
            self.weights[ii] * self._vv[ii][kk - 1] + self._phi_fn(self._zz[kk])[1] - self._phi_fn(self._zz[kk - 1])[1]
        )

        self._ss[ii][kk] += self.weights[ii] * self._ss[ii][kk - 1]
        self._vv[ii][kk] += self.weights[ii] * self._vv[ii][kk - 1]

        for jj in self.neighbors:
            weight = self.weights[jj]
            self._ss[ii][kk][:] += weight * self._ss[jj][kk - 1][:]
            self._vv[ii][kk][:] += weight * self._vv[jj][kk - 1][:]

        cost = self._cost_fn(self.target_position, self._zz[kk], self._ss[ii][kk])[0]
        total_grad = self._cost_fn(self.target_position, self._zz[kk], self._ss[ii][kk])[1:]
        return cost, np.linalg.norm(total_grad)

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
    np.random.seed(0)
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
