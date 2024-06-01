import sys

import numpy as np
import rclpy
import wandb
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

        self._info(
            f"Starting warden {self.id} at position x = {self.pose.position.x}, y = {self.pose.position.y} with target x = {self.target.position.x}, y = {self.target.position.y}"
        )

        self._subs = {}
        for neighbor in self.neighbors:
            self._info(f"Subscribing to neighbor {neighbor}")
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
        self._ss = [{jj: np.zeros(2) for jj in self.neighbors}]
        self._vv = [{jj: np.zeros(2) for jj in self.neighbors}]

        self._ss[self._simtime][self.id] = self._phi_fn(self._zz[self._simtime])[0]
        self._vv[self._simtime][self.id] = self._cost_fn(
            self.target_position, self._zz[self._simtime], self._ss[self._simtime][self.id]
        )[2]

    def _neighbor_callback(self, msg):
        neighbor_id = msg.warden_id
        ss = msg.ss
        vv = msg.vv

        if self._simtime >= len(self._ss):
            self._ss.append({})
            self._vv.append({})

        self._ss[self._simtime][neighbor_id] = np.array(ss)
        self._vv[self._simtime][neighbor_id] = np.array(vv)
        self._info(f"Received data from neighbor {neighbor_id}")

    def _timer_callback(self):
        msg = NodeData()
        msg.warden_id = self.id

        if self._simtime == 0:
            msg.ss = [float(self._ss[self._simtime][self.id][0]), float(self._ss[self._simtime][self.id][1])]
            msg.vv = [float(self._vv[self._simtime][self.id][0]), float(self._vv[self._simtime][self.id][1])]
            self._publisher.publish(msg)
            self._info("Published initial data")
            self._simtime += 1
            return

        if self._simtime >= len(self._ss):
            return

        all_received = all(neighbor_id in self._ss[self._simtime] for neighbor_id in self.neighbors)
        if not all_received:
            self._info("Waiting for all neighbors to respond...")
            return

        self._info("All neighbors have responded. Updating pose...")
        cost, grad = self._update()
        msg.ss = [float(self._ss[self._simtime][self.id][0]), float(self._ss[self._simtime][self.id][1])]
        msg.vv = [float(self._vv[self._simtime][self.id][0]), float(self._vv[self._simtime][self.id][1])]
        self._publisher.publish(msg)
        self._info("Published updated data")
        # wandb.log({"cost": cost, "grad": grad, "zz": self._zz[self._simtime], "id": self.id})

        data = PlotterData()
        data.warden_id = self.id
        data.zz = [float(self._zz[self._simtime][0]), float(self._zz[self._simtime][1])]
        data.cost = cost
        data.grad = grad
        self._plotter_pub.publish(data)
        self._info("Published data for plotter")

        self._simtime += 1

    def _update(self):
        self._info("Updating...")
        kk = self._simtime
        ii = self.id

        li_nabla_1 = self._cost_fn(self.target_position, self._zz[kk - 1], self._ss[kk - 1][ii])[1]
        _, phi_grad = self._phi_fn(self._zz[kk - 1])

        self._zz.append(self._zz[kk - 1] - self.alpha * (li_nabla_1 + self._vv[kk - 1][ii] + phi_grad))

        self._ss[kk][ii] += self._phi_fn(self._zz[kk])[0] - self._phi_fn(self._zz[kk - 1])[0]
        self._vv[kk][ii] += self._phi_fn(self._zz[kk])[1] - self._phi_fn(self._zz[kk - 1])[1]

        self._info(f"Updating from self with weight {self.weights[ii]}")
        self._ss[kk][ii] += self.weights[ii] * self._ss[kk - 1][ii]
        self._vv[kk][ii] += self.weights[ii] * self._vv[kk - 1][ii]
        for jj in self.neighbors:
            weight = self.weights[jj]
            self._info(f"Updating from neighbor {jj} with weight {weight}")
            self._ss[kk][ii] += weight * self._ss[kk - 1][jj]
            self._vv[kk][ii] += weight * self._vv[kk - 1][jj]

        cost = self._cost_fn(self.target_position, self._zz[kk], self._ss[kk][ii])[0]
        total_grad = self._cost_fn(self.target_position, self._zz[kk], self._ss[kk][ii])[1:]
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
    rclpy.init(args=args)
    # wandb.init(project="surveillance")

    try:
        warden = Warden()
        rclpy.spin(warden)
    except KeyboardInterrupt:
        pass
    except BaseException:
        print("Exception in warden:", file=sys.stderr)
        raise
    finally:
        # wandb.finish()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
