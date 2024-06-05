import networkx as nx
import numpy as np


class NodeUpdater:
    def __init__(self, node_index, cost_fn, phi_fn, neighbors, alpha):
        self.node_index = node_index
        self.cost_fn = cost_fn
        self.phi_fn = phi_fn
        self.alpha = alpha

        # NOTE: neighbors is a dictionary with the neighbor index as key and the weight as value and it also includes the node itself
        self.neighbors = neighbors

    def update(self, kk, target, zz, ss, vv):
        ii = self.node_index

        li_nabla_1 = self.cost_fn(target, zz[kk - 1, ii, :], ss[kk - 1, ii, :])[1]
        _, phi_grad = self.phi_fn(zz[kk - 1, ii, :])

        zz[kk, ii, :] = zz[kk - 1, ii, :] - self.alpha * (li_nabla_1 + vv[kk - 1, ii, :] + phi_grad)

        ss[kk, ii, :] += self.phi_fn(zz[kk, ii, :])[0] - self.phi_fn(zz[kk - 1, ii, :])[0]
        vv[kk, ii, :] += self.phi_fn(zz[kk, ii, :])[1] - self.phi_fn(zz[kk - 1, ii, :])[1]

        for jj, weight in self.neighbors.items():
            ss[kk, ii, :] += weight * ss[kk - 1, jj, :]
            vv[kk, ii, :] += weight * vv[kk - 1, jj, :]

        return zz, ss, vv


class AggregativeTracking:
    def __init__(self, cost_fn, phi_fn, max_iters=1000, alpha=1e-2):
        self.cost_fn = cost_fn
        self.phi_fn = phi_fn
        self.max_iters = max_iters
        self.alpha = alpha

        self.cost = np.zeros(max_iters)
        self.gradient_magnitude = np.zeros(max_iters)
        self.node_updaters = []

    def run(self, graph, targets, d):
        nn = len(nx.nodes(graph))
        Adj = nx.adjacency_matrix(graph).toarray()

        AA = np.zeros(shape=(nn, nn))

        # Metropolis-Hastings algorithm
        for ii in range(nn):
            N_ii = np.nonzero(Adj[ii])[0]
            deg_ii = len(N_ii)
            for jj in N_ii:
                deg_jj = len(np.nonzero(Adj[jj])[0])
                AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

        AA += np.eye(nn) - np.diag(np.sum(AA, axis=0))

        zz = np.zeros((self.max_iters, nn, d))  # * 10 - 5
        ss = np.zeros((self.max_iters, nn, d))
        vv = np.zeros((self.max_iters, nn, d))

        for ii in range(nn):
            ss[0, ii, :] = self.phi_fn(zz[0, ii, :])[0]
            vv[0, ii, :] = self.cost_fn(targets[ii], zz[0, ii, :], ss[0, ii, :])[2]

            neighbors = {jj: AA[ii, jj] for jj in list(graph.neighbors(ii)) + [ii]}
            updater = NodeUpdater(
                ii,
                self.cost_fn,
                self.phi_fn,
                neighbors,
                self.alpha,
            )
            self.node_updaters.append(updater)

        for kk in range(1, self.max_iters):
            for ii in range(nn):
                target = targets[ii]
                zz, ss, vv = self.node_updaters[ii].update(kk, target, zz, ss, vv)

                total_grad = self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[1:]
                grad = np.linalg.norm(total_grad)
                self.gradient_magnitude[kk] += grad

                cost = self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[0]
                self.cost[kk] += cost

            print(f"Iteration: #{kk}, Cost: {self.cost[kk]:.2f}, Gradient Magnitude: {self.gradient_magnitude[kk]:.2f}")

            # Take the gradient magnitude from the last 10 iterations and check if it's not changing a lot, then stop
            if kk > 10 and np.std(self.gradient_magnitude[kk - 10 : kk]) < 1e-4:
                print("Converged")
                break

        return zz[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk], kk
