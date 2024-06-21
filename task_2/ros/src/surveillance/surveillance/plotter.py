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
        zz_init = self.get_parameter("zz_init").value
        self.zz_init = np.array(zz_init).reshape(self.nodes, 2)
        self.timer_period = self.get_parameter("timer_period").value or DEFAULT_TIMER_PERIOD
        Adj = self.get_parameter("Adj").value
        self.Adj = np.array(Adj).reshape(self.nodes, self.nodes)
        self.max_iters = self.get_parameter("max_iters").value
        self.cost_type = self.get_parameter("cost_type").value

        self._debug("Start animation")
        self._sub = self.create_subscription(PlotterData, "/plotter", self._node_callback, 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self.finished = False

        self._simtime = 0
        self._node_costs = {ii: [] for ii in range(self.nodes)}
        self._node_grads = {ii: [] for ii in range(self.nodes)}
        self._zz = {ii: [] for ii in range(self.nodes)}

        self._cost = np.zeros(self.max_iters)
        self._grad = np.zeros(self.max_iters)

    def _node_callback(self, msg):
        node_id = msg.warden_id
        zz = msg.zz
        cost = msg.cost
        grad = msg.grad
        time = msg.time

        if self.finished:
            return

        self._zz[node_id].append(np.array(zz))
        self._debug(f"Received data from node {node_id} (#{time})")

        if time > 0:
            self._node_costs[node_id].append(cost)
            self._node_grads[node_id].append(np.array(grad))

    def _timer_callback(self):
        kk = self._simtime

        all_received = all(len(self._node_costs[node_id]) > self._simtime for node_id in range(self.nodes))
        if not all_received:
            self._debug("Waiting for all neighbors to respond...")
            return

        self._cost[kk] = sum(self._node_costs[node_id][kk] for node_id in range(self.nodes))
        self._grad[kk] = np.linalg.norm(sum(self._node_grads[node_id][kk] for node_id in range(self.nodes)))

        self._info(f"Iteration: #{kk}, Cost: {self._cost[kk]:.2f}, Gradient Magnitude: {self._grad[kk]:.2f}")

        if kk == self.max_iters - 1 or self._grad[kk] < 1e-6:
            self.finished = True
            self._timer.destroy()

            # Convert zz to numpy array
            zz = np.zeros((kk, self.nodes, 2))
            for ii in range(self.nodes):
                zz[:, ii, :] = np.array(self._zz[ii])[:kk, :]

            self._info(f"Results: {zz[-1, :]}")

            _, ax = plt.subplots(1, 2, figsize=(10, 10))

            ax[0].semilogy(np.arange(zz.shape[0] - 1), self._cost[: kk - 1])
            ax[0].grid()
            ax[0].set_title("Cost")

            ax[1].semilogy(np.arange(zz.shape[0] - 1), self._grad[: kk - 1])
            ax[1].grid()
            ax[1].set_title("Gradient magnitude")
            plt.show()

            for jj in range(self.nodes):
                plt.plot(
                    zz[:, jj, 0],
                    zz[:, jj, 1],
                    linewidth=1,
                    color="black",
                    linestyle="dashed",
                    label=f"Trajectory {jj}",
                )

                plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", marker="x")

                plt.plot(self.targets[:, 0], self.targets[:, 1], "bx")
                plt.plot(self.zz_init[:, 0], self.zz_init[:, 1], "ro")

                if self.cost_type == "corridor":
                    x = np.linspace(-60, 60, 100)
                    g_1 = 1e-5 * x**4 + 2
                    g_2 = -(1e-5 * x**4 + 2)
                    plt.plot(x, g_1, "g-")
                    plt.plot(x, g_2, "g-")

                plt.title("Trajectories")

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
