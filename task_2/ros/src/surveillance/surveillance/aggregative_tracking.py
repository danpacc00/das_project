import networkx as nx
import numpy as np


class AggregativeTracking:
    def __init__(self, cost_fn, phi_fn, max_iters=1000, alpha=1e-2):
        self.cost_fn = cost_fn
        self.phi_fn = phi_fn
        self.max_iters = max_iters
        self.initial_alpha = alpha
        self.alpha = alpha

        self.cost = np.zeros(max_iters)
        self.gradient_magnitude = np.zeros(max_iters)

    def run(self, graph, initial_poses, targets, d):
        nn = len(nx.nodes(graph))
        Adj = nx.adjacency_matrix(graph).toarray()

        AA = np.zeros(shape=(nn, nn))

        # Metropolis-Hastings weights in order to have a doubly stochastic matrix
        for ii in range(nn):
            N_ii = np.nonzero(Adj[ii])[0]
            deg_ii = len(N_ii)
            for jj in N_ii:
                deg_jj = len(np.nonzero(Adj[jj])[0])
                AA[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))

        AA += np.eye(nn) - np.diag(np.sum(AA, axis=0))

        zz = np.zeros((self.max_iters, nn, d))
        for ii in range(nn):
            zz[0, ii, :] = initial_poses[ii]

        ss = np.zeros((self.max_iters, nn, d))
        vv = np.zeros((self.max_iters, nn, d))

        for ii in range(nn):
            ss[0, ii, :] = self.phi_fn(zz[0, ii, :])[0]
            vv[0, ii, :] = self.cost_fn(targets[ii], zz[0, ii, :], ss[0, ii, :])[2]

        self.l_nabla_2 = np.zeros((self.max_iters, d))

        for kk in range(self.max_iters - 1):
            grad = np.zeros((2 * d))

            for ii in range(nn):
                N_ii = np.nonzero(Adj[ii])[0]
                ss[kk + 1, ii, :] += AA[ii, ii] * ss[kk, ii, :]
                vv[kk + 1, ii, :] += AA[ii, ii] * vv[kk, ii, :]
                for jj in N_ii:
                    ss[kk + 1, ii, :] += AA[ii, jj] * ss[kk, jj, :]
                    vv[kk + 1, ii, :] += AA[ii, jj] * vv[kk, jj, :]

                target = targets[ii]
                li_nabla_1 = self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[1]

                _, phi_grad = self.phi_fn(zz[kk, ii, :])

                if "constraints" in dir(self.cost_fn):
                    constraints = self.cost_fn.constraints(zz[kk, ii, :])
                    if np.any(constraints**2 <= 1.0):
                        self.alpha = np.max([self.alpha * 1e-3, 5e-5])
                    else:
                        self.alpha = np.min([self.alpha * 1.1, self.initial_alpha])

                    zz[kk + 1, ii, :] = zz[kk, ii, :] - self.alpha * (li_nabla_1 + phi_grad * vv[kk, ii, :])
                else:
                    zz[kk + 1, ii, :] = zz[kk, ii, :] - self.alpha * (li_nabla_1 + phi_grad * vv[kk, ii, :])

                ss[kk + 1, ii, :] += self.phi_fn(zz[kk + 1, ii, :])[0] - self.phi_fn(zz[kk, ii, :])[0]
                vv[kk + 1, ii, :] += (
                    self.cost_fn(target, zz[kk + 1, ii, :], ss[kk + 1, ii, :])[2]
                    - self.cost_fn(target, zz[kk, ii, :], ss[kk, ii, :])[2]
                )

                cost, li_nabla_1, li_nabla_2 = self.cost_fn(target, zz[kk + 1, ii, :], ss[kk + 1, ii, :])
                # grad += li_nabla_1 + li_nabla_2  # TODO: fix this
                grad[:d] += li_nabla_1
                grad[d:] += li_nabla_2

                self.cost[kk] += cost

                self.l_nabla_2[kk] += li_nabla_2

            self.gradient_magnitude[kk] += np.linalg.norm(grad)

            print(f"Iteration: #{kk}, Cost: {self.cost[kk]:.2f}, Gradient Magnitude: {self.gradient_magnitude[kk]:.2f}")
            if self.gradient_magnitude[kk] < 1e-6:
                print(f"Converged at iteration {kk}")
                break
        # Evaluate the difference between vv and l_nabla_2
        v_nabla2_diff = np.zeros((self.max_iters, nn))
        for ii in range(nn):
            v_nabla2_diff[:, ii] = np.linalg.norm(vv[:, ii, :] - self.l_nabla_2, axis=1)

        return zz[:kk, :, :], ss[:kk, :, :], self.cost[:kk], self.gradient_magnitude[:kk], v_nabla2_diff[:kk]
