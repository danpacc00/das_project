import networkx as nx
import numpy as np


class AggregativeTracking:
    def __init__(self, cost_fn, phi_fn, max_iters=1000, alpha=1e-2):
        self.cost_fn = cost_fn
        self.phi_fn = phi_fn
        self.max_iters = max_iters
        self.alpha = alpha

        self.cost = np.zeros(max_iters)
        self.gradient_magnitude = np.zeros(max_iters)
        self.node_updaters = []

    def run(self, graph, initial_poses, targets, d):
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

        zz = np.zeros((self.max_iters, nn, d))
        zz[0, :, :] = initial_poses

        ss = np.zeros((self.max_iters, nn, d))
        vv = np.zeros((self.max_iters, nn, d))

        for ii in range(nn):
            ss[0, ii, :] = self.phi_fn(zz[0, ii, :])[0]
            vv[0, ii, :] = self.cost_fn(targets[ii], zz[0, ii, :], ss[0, ii, :])[2]

        for kk in range(1, self.max_iters):
            grad = np.zeros((d, d))

            for ii in range(nn):
                target = targets[ii]
                li_nabla_1 = self.cost_fn(target, zz[kk - 1, ii, :], ss[kk - 1, ii, :])[1]
                _, phi_grad = self.phi_fn(zz[kk - 1, ii, :])

                zz[kk, ii, :] = zz[kk - 1, ii, :] - self.alpha * (li_nabla_1 + phi_grad * vv[kk - 1, ii, :])

                ss[kk, ii, :] += self.phi_fn(zz[kk, ii, :])[0] - self.phi_fn(zz[kk - 1, ii, :])[0]
                vv[kk, ii, :] += (
                    self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[2]
                    - self.cost_fn(target, zz[kk - 1, ii, :], ss[kk - 1, ii, :])[2]
                )

                N_ii = np.nonzero(Adj[ii])[0]
                for jj in N_ii:
                    ss[kk, ii, :] += AA[ii, jj] * ss[kk - 1, jj, :]
                    vv[kk, ii, :] += AA[ii, jj] * vv[kk - 1, jj, :]

                grad += self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[1:]
                cost = self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[0]
                self.cost[kk] += cost

            self.gradient_magnitude[kk] += np.linalg.norm(grad)

            print(f"Iteration: #{kk}, Cost: {self.cost[kk]:.2f}, Gradient Magnitude: {self.gradient_magnitude[kk]:.2f}")

        return zz[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk], kk
