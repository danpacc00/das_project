import sys

import matplotlib.pyplot as plt
import numpy as np
import rclpy
from message.msg import PlotterData
from rclpy.node import Node

from surveillance.functions import simple_animation, corridor_animation

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
        self.tradeoff = self.get_parameter("tradeoff").value

        self._debug("Start animation")
        self._sub = self.create_subscription(PlotterData, "/plotter", self._node_callback, 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self.finished = False

        self._simtime = 0
        self._node_costs = {ii: [] for ii in range(self.nodes)}
        self._node_grads = {ii: [] for ii in range(self.nodes)}
        self._zz = {ii: [] for ii in range(self.nodes)}
        self._ss = {ii: [] for ii in range(self.nodes)}

        self._cost = np.zeros(self.max_iters)
        self._grad = np.zeros(self.max_iters)

    def _node_callback(self, msg):
        node_id = msg.warden_id
        zz = msg.zz
        ss = msg.ss
        cost = msg.cost
        grad = msg.grad
        time = msg.time

        # Warden nodes keep sending data even after the simulation has finished
        # so we need to ignore any data received after the simulation has ended
        if self.finished:
            return

        self._zz[node_id].append(np.array(zz))
        self._ss[node_id].append(np.array(ss))
        self._debug(f"Received data from node {node_id} (#{time})")

        # Initial cost and gradient values are meaningless (they are hardcoded as zero, see warden.py code), 
        # so we ignore them
        if time > 0:
            self._node_costs[node_id].append(cost)
            self._node_grads[node_id].append(np.array(grad))

    def _timer_callback(self):
        kk = self._simtime

        # Check if all nodes have sent their data for the current iteration
        all_received = all(len(self._node_costs[node_id]) > self._simtime for node_id in range(self.nodes))
        if not all_received:
            self._debug("Waiting for all neighbors to respond...")
            return

        # Calculate the total cost and gradient magnitude for the current iteration
        self._cost[kk] = sum(self._node_costs[node_id][kk] for node_id in range(self.nodes))
        self._grad[kk] = np.linalg.norm(sum(self._node_grads[node_id][kk] for node_id in range(self.nodes)))

        self._info(f"Iteration: #{kk}, Cost: {self._cost[kk]:.2f}, Gradient Magnitude: {self._grad[kk]:.2f}")

        if kk == self.max_iters - 1 or self._grad[kk] < 1e-6:
            self.finished = True
            self._timer.destroy()

            # Convert the lists of lists to numpy arrays
            zz = np.zeros((kk, self.nodes, 2))
            for ii in range(self.nodes):
                zz[:, ii, :] = np.array(self._zz[ii])[:kk, :]

            # Convert the lists of lists to numpy arrays
            ss = np.zeros((kk, self.nodes, 2))
            for ii in range(self.nodes):
                ss[:, ii, :] = np.array(self._ss[ii])[:kk, :]

            if self.cost_type == "surveillance":
                self._surveillance_plots(ss, zz, kk)
            else:
                self._corridor_plots(ss, zz, kk)

        self._simtime += 1

    def _corridor_plots(self, ss, zz, kk):
        top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}
        bottom_wall = {"x_start": -15, "x_end": 15, "y": -5, "res": 1000}
        y_offset = 20

        x = np.linspace(-60, 60, 100)
        g_1 = 1e-5 * x**4 + 2
        g_2 = -(1e-5 * x**4 + 2)

        _, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].semilogx(np.arange(ss.shape[0]), ss[:, :, 0])
        ax[0].grid()
        ax[0].set_title("$s_x$")

        ax[1].semilogx(np.arange(ss.shape[0]), ss[:, :, 1])
        ax[1].grid()
        ax[1].set_title("$s_y$")
        plt.suptitle("Barycenter estimation (obstacles case)")
        plt.show()

        _, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].semilogy(np.arange(zz.shape[0] - 1), self._cost[: kk - 1])
        ax[0].grid()
        ax[0].set_title("Cost")

        ax[1].semilogy(np.arange(zz.shape[0] - 1), self._grad[1:kk])
        ax[1].grid()
        ax[1].set_title("Gradient magnitude")
        plt.suptitle(f"Aggregative tracking with tradeoff = {self.tradeoff}")
        plt.show()

        # plot trajectories
        for jj in range(self.nodes):
            plt.plot(
                zz[:, jj, 0],
                zz[:, jj, 1],
                linewidth=1,
                color="black",
                linestyle="dashed",
                label=f"Trajectory {jj}",
            )

            plt.scatter(zz[-1, jj, 0], zz[-1, jj, 1], color="orange", label=f"Final position {jj}", marker="x")

            plt.annotate(
                f"$z_{jj}^0$",
                xy=(zz[0, jj, 0], zz[0, jj, 1]),
                xytext=(zz[0, jj, 0] - 2.0, zz[0, jj, 1] + 3.5),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
            )

            plt.plot(self.targets[:, 0], self.targets[:, 1], "bx")
            plt.plot(self.zz_init[:, 0], self.zz_init[:, 1], "ro")

            plt.plot(
                np.linspace(top_wall["x_start"], top_wall["x_end"], top_wall["res"]),
                np.tile(top_wall["y"], top_wall["res"]),
                "k",
            )
            plt.plot(
                np.linspace(bottom_wall["x_start"], bottom_wall["x_end"], bottom_wall["res"]),
                np.tile(bottom_wall["y"], bottom_wall["res"]),
                "k",
            )
            plt.plot(
                np.tile(top_wall["x_start"], top_wall["res"]),
                np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
                "k",
            )
            plt.plot(
                np.tile(bottom_wall["x_start"], bottom_wall["res"]),
                np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
                "k",
            )
            plt.plot(
                np.tile(top_wall["x_end"], top_wall["res"]),
                np.linspace(top_wall["y"], top_wall["y"] + y_offset * 4, top_wall["res"]),
                "k",
            )
            plt.plot(
                np.tile(bottom_wall["x_end"], bottom_wall["res"]),
                np.linspace(bottom_wall["y"], bottom_wall["y"] - y_offset * 4, bottom_wall["res"]),
                "k",
            )

            plt.annotate(
                f"Target {jj}",
                xy=(self.targets[jj, 0], self.targets[jj, 1]),
                xytext=(self.targets[jj, 0] + 3.5, self.targets[jj, 1] + 2.5),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="blue"),
            )

            plt.title("Agents trajectories")

            plt.plot(x, g_1, color="green", linestyle="dashed")
            plt.plot(x, g_2, color="green", linestyle="dashed")

        plt.ylim(-60, 60)
        plt.show()

        corridor_animation(
            zz,
            np.linspace(0, kk, kk),
            self.Adj,
            self.targets,
            top_wall,
            bottom_wall,
            y_offset,
        )

    def _surveillance_plots(self, ss, zz, kk):
        _, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].semilogx(np.arange(ss.shape[0]), ss[:, :, 0])
        ax[0].grid()
        ax[0].set_title("$s_x$")

        ax[1].semilogx(np.arange(ss.shape[0]), ss[:, :, 1])
        ax[1].grid()
        ax[1].set_title("$s_y$")
        plt.suptitle(f"Barycenter estimation with tradeoff = {self.tradeoff}")
        plt.show()

        _, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].semilogy(np.arange(zz.shape[0] - 1), self._cost[: kk - 1])
        ax[0].grid()
        ax[0].set_title("Cost")

        ax[1].semilogy(np.arange(zz.shape[0] - 1), self._grad[1:kk])
        ax[1].grid()
        ax[1].set_title("Gradient magnitude")
        plt.suptitle(f"Aggregative tracking with tradeoff = {self.tradeoff}")
        plt.show()

        # plot trajectories
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

            plt.annotate(
                f"$z_{jj}^0$",
                xy=(zz[0, jj, 0], zz[0, jj, 1]),
                xytext=(zz[0, jj, 0] + 0.2, zz[0, jj, 1] + 0.2),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="red"),
            )

            plt.plot(self.targets[:, 0], self.targets[:, 1], "bx")
            plt.plot(self.zz_init[:, 0], self.zz_init[:, 1], "ro")

            if self.tradeoff == 10.0:
                label_offsets = [(0.2, 0.2), (-0.2, -0.7), (-0.8, -0.7), (0.2, -0.2)]
            else:
                label_offsets = [(0.1, 0.2), (0.1, 0.2), (-0.55, -0.35), (-0.55, -0.35)]

            plt.annotate(
                f"Target {jj}",
                xy=(self.targets[jj, 0], self.targets[jj, 1]),
                xytext=(self.targets[jj, 0] + label_offsets[jj][0], self.targets[jj, 1] + label_offsets[jj][1]),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="blue"),
            )

            plt.title(f"Agents trajectories (tradeoff = {self.tradeoff})")

        plt.show()

        simple_animation(zz, np.linspace(0, kk, kk), self.Adj, self.targets)

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
