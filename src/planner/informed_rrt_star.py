import numpy as np
from planner.rrt_star import RRTStar, Node
from common.utils import get_dist, is_collision_free

class InformedRRTStar(RRTStar):
    def __init__(self, start, goal, obstacles, bounds,
                 step_size=0.5, max_iter=1000, search_radius=1.5,
                 rng=None,
                 goal_sample_rate=0.0,   # typically lower because informed sampling dominates
                 goal_connect_radius=5.0,
                 goal_tolerance=2.0):
        super().__init__(
            start, goal, obstacles, bounds,
            step_size=step_size, max_iter=max_iter, search_radius=search_radius,
            rng=rng,
            goal_sample_rate=goal_sample_rate,
            goal_connect_radius=goal_connect_radius,
            goal_tolerance=goal_tolerance
        )
        self.c_best = float("inf")

    def step(self):
        """One informed RRT* iteration."""
        self.iterations += 1

        rnd_node = self.get_informed_state()
        nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
        nearest_node = self.node_list[nearest_ind]

        new_node = self.steer(nearest_node, rnd_node, self.step_size)
        if new_node is None:
            return

        if new_node.parent is not None and not is_collision_free(new_node.parent.state, new_node.state, self.obstacles):
            return

        near_inds = self.find_near_nodes(new_node)
        new_node = self.choose_parent(new_node, near_inds)

        if new_node:
            self.node_list.append(new_node)
            self.rewire(new_node, near_inds)

            # update informed bound if "close enough" to goal
            dist_to_goal = self.calc_dist_to_goal(new_node.state)
            if dist_to_goal <= self.step_size:
                cand = new_node.cost + dist_to_goal
                if cand < self.c_best:
                    self.c_best = cand

            # same explicit goal-connection as base (fairness)
            if dist_to_goal <= self.goal_connect_radius:
                if is_collision_free(new_node.state, self.goal.state, self.obstacles):
                    goal_node = Node(self.goal.state)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    self.node_list.append(goal_node)

                    if goal_node.cost < self.best_cost:
                        self.best_cost = goal_node.cost
                        self.best_goal_node = goal_node
                        if goal_node.cost < self.c_best:
                            self.c_best = goal_node.cost

    def get_informed_state(self):
        if self.c_best == float("inf"):
            return self.get_random_node()

        c_min = get_dist(self.start.state, self.goal.state)
        x_start = self.start.state
        x_goal = self.goal.state
        x_center = (x_start + x_goal) / 2.0

        if self.c_best < c_min:
            self.c_best = c_min

        C = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), (x_goal - x_start) / c_min)

        L = np.diag([
            self.c_best / 2.0,
            np.sqrt(self.c_best**2 - c_min**2) / 2.0,
            np.sqrt(self.c_best**2 - c_min**2) / 2.0
        ])

        while True:
            xs = self.sample_unit_ball()
            x_rand = (C @ L @ xs) + x_center
            if np.all(x_rand >= self.bounds[:, 0]) and np.all(x_rand <= self.bounds[:, 1]):
                return Node(x_rand)

    def sample_unit_ball(self):
        while True:
            if self.rng is None:
                x = np.random.uniform(-1, 1, 3)
            else:
                x = self.rng.uniform(-1, 1, 3)
            if np.linalg.norm(x) <= 1:
                return x

    def rotation_matrix_from_vectors(self, vec1, vec2):
        a = (vec1 / np.linalg.norm(vec1)).reshape(3)
        b = (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if np.any(v):
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return np.eye(3)
