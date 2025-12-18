import numpy as np
from common.utils import is_collision_free, sample_uniform

class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.g_score = np.inf
        self.f_score = np.inf
        self.children = []

class BITStar:
    def __init__(self, start, goal, obstacles, bounds, eta=1.1, batch_size=100, rng=None):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.eta = eta
        self.batch_size = batch_size
        self.rng = rng

        self.start_node = Node(self.start)
        self.goal_node = Node(self.goal)

        self.V = []
        self.X_samples = []
        self.QE = []
        self.QV = []

        self.r = np.inf
        self.g_goal = np.inf

        self.iterations = 0

        self._initialized = False

    def get_heuristic(self, state):
        return float(np.linalg.norm(state - self.goal))

    def initialize(self):
        self.start_node.g_score = 0.0
        self.start_node.f_score = self.get_heuristic(self.start)
        self.V = [self.start_node]
        self.X_samples = []
        self.QE = []
        self.QV = []
        self.r = np.inf
        self.g_goal = np.inf
        self.goal_node = Node(self.goal)
        self._initialized = True

    def step(self):
        """One BIT* expansion step (one vertex expansion + one edge attempt at most)."""
        if not self._initialized:
            self.initialize()

        self.iterations += 1

        if not self.QE and not self.QV:
            self.prune()
            self.X_samples.extend(self.sample_batch(self.batch_size))

            q = len(self.V) + len(self.X_samples)
            dim = len(self.start)
            if q > 1:
                self.r = self.eta * 5.0 * (np.log(q) / q) ** (1 / dim)
            else:
                self.r = np.inf

            self.QV = [v for v in self.V if v.f_score < self.g_goal]

        # expand one vertex into candidate edges
        if self.QV:
            v_curr = self.QV.pop(0)
            for x_state in list(self.X_samples):
                dist = float(np.linalg.norm(v_curr.state - x_state))
                if dist <= self.r:
                    g_est = v_curr.g_score + dist + self.get_heuristic(x_state)
                    if g_est < self.g_goal:
                        self.QE.append((v_curr, x_state, dist))

        if self.QE:
            self.QE.sort(key=lambda e: e[0].g_score + e[2] + self.get_heuristic(e[1]))
            v, x_state, dist = self.QE.pop(0)

            if is_collision_free(v.state, x_state, self.obstacles):
                new_node = Node(x_state)
                new_node.parent = v
                new_node.g_score = v.g_score + dist
                new_node.f_score = new_node.g_score + self.get_heuristic(x_state)

                # goal check
                if np.linalg.norm(new_node.state - self.goal) <= 1.0:
                    if new_node.g_score < self.g_goal:
                        self.g_goal = new_node.g_score
                        self.goal_node = new_node

                self.V.append(new_node)
                self.QV.append(new_node)

                # remove sample
                for i, s in enumerate(self.X_samples):
                    if np.array_equal(s, x_state):
                        self.X_samples.pop(i)
                        break

    def sample_batch(self, k):
        samples = []
        if self.g_goal < np.inf:
            c_min = float(np.linalg.norm(self.goal - self.start))
            c_best = float(self.g_goal)
            if c_best < c_min:
                c_best = c_min

            x_center = (self.start + self.goal) / 2.0
            a1 = (self.goal - self.start) / c_min
            C = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), a1)
            L = np.diag([c_best / 2.0,
                         np.sqrt(c_best**2 - c_min**2) / 2.0,
                         np.sqrt(c_best**2 - c_min**2) / 2.0])

            for _ in range(k):
                while True:
                    if self.rng is None:
                        x_ball = np.random.uniform(-1, 1, 3)
                    else:
                        x_ball = self.rng.uniform(-1, 1, 3)
                    if np.linalg.norm(x_ball) <= 1:
                        break

                x_rand = (C @ L @ x_ball) + x_center
                if np.all(x_rand >= self.bounds[:, 0]) and np.all(x_rand <= self.bounds[:, 1]):
                    samples.append(x_rand)
                    if len(samples) >= k:
                        break
        else:
            for _ in range(k):
                samples.append(sample_uniform(self.bounds, rng=self.rng))
        return samples

    def rotation_matrix_from_vectors(self, vec1, vec2):
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        if np.any(v):
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return np.eye(3)

    def prune(self):
        self.V = [v for v in self.V if v.f_score < self.g_goal]
        self.X_samples = [x for x in self.X_samples if self.get_heuristic(x) < self.g_goal]

    def reconstruct_path(self):
        if self.g_goal == np.inf:
            return None
        path = []
        cur = self.goal_node
        while cur:
            path.append(cur.state)
            cur = cur.parent
        return path[::-1]

    def get_best_solution(self):
        path = self.reconstruct_path()
        if path is None:
            return None, False, float("inf")
        path = np.asarray(path)
        length = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))) if len(path) >= 2 else 0.0
        # objective for BIT* is g_goal (path cost)
        return path, True, float(self.g_goal if self.g_goal < np.inf else length)

    @property
    def nodes_in_tree(self):
        return len(self.V)
