import sys

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from message.msg import PlotterData
from rclpy.node import Node

from surveillance.functions import animation

DEFAULT_TIMER_PERIOD = 2  # seconds


class Plotter(Node):
    def __init__(self):
        super().__init__(
            "warden", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True
        )

        self.id = self.get_parameter("id").value
        self.nodes = self.get_parameter("nodes").value
        targets = self.get_parameter("targets").value
        self.targets = np.array(targets).reshape(self.nodes, 2)
        self.timer_period = self.get_parameter("timer_period").value or DEFAULT_TIMER_PERIOD
        Adj = self.get_parameter("Adj").value
        self.Adj = np.array(Adj).reshape(self.nodes, self.nodes)

        self._info("Start animation")
        self._sub = self.create_subscription(PlotterData, "/plotter", self._node_callback, 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)

        self._simtime = 0
        self._cost = []
        self._grad = []
        self._zz = []

    def _node_callback(self, msg):
        node_id = msg.warden_id
        zz = msg.zz
        cost = msg.cost
        grad = msg.grad

        if self._simtime >= len(self._zz):
            self._zz.append(np.empty((self.nodes, 2), dtype=object))
            self._cost.append(0)
            self._grad.append(0)

        self._cost[self._simtime] += cost
        self._grad[self._simtime] += grad
        self._zz[self._simtime][node_id] = np.array(zz)
        self._info(f"Received data from node {node_id}")

    def _timer_callback(self):
        kk = self._simtime
        if kk >= len(self._zz):
            return

        all_received = all((self._zz[kk][node_id] != np.empty(2, dtype=object)).all() for node_id in range(self.nodes))
        if not all_received:
            self._info("Waiting for all nodes to respond...")
            return

        self._info(f"Iteration: #{kk}, Cost: {self._cost[kk]:.2f}, Gradient Magnitude: {self._grad[kk]:.2f}")

        self._info("All neighbors have responded. Checking convergency...")
        if kk > 10 and np.std(self._grad[kk - 10 : kk]) < 1e-4 or kk > 150:
            self._info("Convergency reached. Stopping...")
            self._timer.destroy()

            zz = np.array(self._zz)

            _, ax = plt.subplots(3, 1, figsize=(10, 10))
            ax[0].plot(np.arange(zz.shape[0]), zz[:, :, 0])
            ax[0].grid()
            ax[0].set_title("Aggregative tracking")

            ax[1].plot(np.arange(zz.shape[0] - 1), self._cost[:-1])
            ax[1].grid()
            ax[1].set_title("Cost")

            ax[2].semilogy(np.arange(zz.shape[0] - 1), self._grad[1:])
            ax[2].grid()
            ax[2].set_title("Gradient magnitude")
            plt.show()

            plt.figure("Animation")
            animation(zz, np.linspace(0, kk, kk), self.Adj, self.targets)

        self._simtime += 1

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
        plotter = Plotter()
        rclpy.spin(plotter)
    except KeyboardInterrupt:
        pass
    except BaseException:
        print("Exception in Plotter:", file=sys.stderr)
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
