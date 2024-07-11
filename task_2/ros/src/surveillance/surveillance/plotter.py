import sys

import numpy as np
import rclpy
from message.msg import PlotterData
from rclpy.node import Node

import surveillance.plot as plot
from surveillance.functions import corridor_animation, simple_animation

DEFAULT_TIMER_PERIOD = 2  # seconds
DEFAULT_ALPHA = 1e-2  # stepsize


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
        self.alpha = self.get_parameter("alpha").value or DEFAULT_ALPHA

        self._debug("Start animation")
        self._sub = self.create_subscription(PlotterData, "/plotter", self._node_callback, 10)
        self._timer = self.create_timer(self.timer_period, self._timer_callback)
        self.finished = False

        self._simtime = 0
        self._node_costs = {ii: [] for ii in range(self.nodes)}
        self._node_grads = {ii: [] for ii in range(self.nodes)}
        self._node_nabla_2 = {ii: [] for ii in range(self.nodes)}
        self._zz = {ii: [] for ii in range(self.nodes)}
        self._ss = {ii: [] for ii in range(self.nodes)}
        self._vv = {ii: [] for ii in range(self.nodes)}

        self._cost = np.zeros(self.max_iters)
        self._grad = np.zeros(self.max_iters)
        self._total_nabla_2 = np.zeros((self.max_iters, 2))

    def _node_callback(self, msg):
        node_id = msg.warden_id
        zz = msg.zz
        ss = msg.ss
        vv = msg.vv
        cost = msg.cost
        grad = msg.grad
        nabla_2 = msg.nabla_2
        time = msg.time

        # Warden nodes keep sending data even after the simulation has finished
        # so we need to ignore any data received after the simulation has ended
        if self.finished:
            return

        self._zz[node_id].append(np.array(zz))
        self._ss[node_id].append(np.array(ss))
        self._vv[node_id].append(np.array(vv))
        self._debug(f"Received data from node {node_id} (#{time})")

        # Initial cost and gradient values are meaningless (they are hardcoded as zero, see warden.py code),
        # so we ignore them
        if time > 0:
            self._node_costs[node_id].append(cost)
            self._node_grads[node_id].append(np.array(grad))
            self._node_nabla_2[node_id].append(np.array(nabla_2))

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
        self._total_nabla_2[kk] = np.linalg.norm(sum(self._node_nabla_2[node_id][kk] for node_id in range(self.nodes)))

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

            # Convert the lists of lists to numpy arrays
            vv = np.zeros((kk, self.nodes, 2))
            for ii in range(self.nodes):
                vv[:, ii, :] = np.array(self._vv[ii])[:kk, :]

            if self.cost_type == "surveillance":
                self._surveillance_plots(ss, vv, zz, kk)
            else:
                self._corridor_plots(ss, vv, zz, kk)

        self._simtime += 1

    def _corridor_plots(self, ss, vv, zz, kk):
        top_wall = {"x_start": -15, "x_end": 15, "y": 5, "res": 1000}
        bottom_wall = {"x_start": -15, "x_end": 15, "y": -5, "res": 1000}
        y_offset = 20

        x = np.linspace(-60, 60, 100)
        g_1 = 1e-5 * x**4 + 2
        g_2 = -(1e-5 * x**4 + 2)

        barycenter = np.zeros((zz.shape[0], 2))
        for i in range(kk):
            barycenter[i, 0] = np.mean(zz[i, :, 0])  # Mean of x coordinates
            barycenter[i, 1] = np.mean(zz[i, :, 1])  # Mean of y coordinates

        diff_barycenter_s = np.zeros((ss.shape[0], self.nodes))
        v_nabla2_diff = np.zeros((vv.shape[0], self.nodes))
        for ii in range(self.nodes):
            diff_barycenter_s[:, ii] = np.linalg.norm(ss[:, ii, :] - barycenter, axis=1)
            v_nabla2_diff[:, ii] = np.linalg.norm(vv[:, ii, :] - self._total_nabla_2[:kk], axis=1)

        plot.ss_estimates(ss)

        plot.convergence(diff_barycenter_s, v_nabla2_diff)

        plot.cost_gradient(self._cost[: kk - 1], self._grad[1:kk], title_suffix="")

        plot.trajectories(
            zz,
            self.targets,
            self.zz_init,
            2,
            self.alpha,
            additional_elements=[lambda: plot.corridor(top_wall, bottom_wall, y_offset, x, g_1, g_2)],
        )

        corridor_animation(
            zz,
            np.linspace(0, kk, kk),
            self.Adj,
            self.targets,
            top_wall,
            bottom_wall,
            y_offset,
        )

    def _surveillance_plots(self, ss, vv, zz, kk):
        barycenter = np.zeros((zz.shape[0], 2))
        for i in range(kk):
            barycenter[i, 0] = np.mean(zz[i, :, 0])  # Mean of x coordinates
            barycenter[i, 1] = np.mean(zz[i, :, 1])  # Mean of y coordinates

        diff_barycenter_s = np.zeros((ss.shape[0], self.nodes))
        v_nabla2_diff = np.zeros((vv.shape[0], self.nodes))
        for ii in range(self.nodes):
            diff_barycenter_s[:, ii] = np.linalg.norm(ss[:, ii, :] - barycenter, axis=1)
            v_nabla2_diff[:, ii] = np.linalg.norm(vv[:, ii, :] - self._total_nabla_2[:kk], axis=1)

        plot.ss_estimates(ss)

        plot.convergence(diff_barycenter_s, v_nabla2_diff)

        plot.cost_gradient(self._cost[: kk - 1], self._grad[1:kk], title_suffix=f"($\\gamma = {self.tradeoff}$)")

        plot.trajectories(zz, self.targets, self.zz_init, 0, self.tradeoff)

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
