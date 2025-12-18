import numpy as np
from common.utils import get_dist, is_collision_free, sample_uniform

class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, bounds,
                 step_size=0.5, max_iter=1000, search_radius=1.5,
                 rng=None,
                 goal_sample_rate=0.2,
                 goal_connect_radius=5.0,
                 goal_tolerance=2.0):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius

        self.rng = rng  # np.random.Generator preferred
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_connect_radius = float(goal_connect_radius)
        self.goal_tolerance = float(goal_tolerance)

        self.node_list = [self.start]

        # incremental bookkeeping
        self.iterations = 0
        self.best_goal_node = None
        self.best_cost = float("inf")

    # -----------------------
    # Incremental interface
    # -----------------------
    def step(self):
        """Run exactly one RRT* iteration."""
        self.iterations += 1

        rnd_node = self.get_random_node()
        nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
        nearest_node = self.node_list[nearest_ind]

        new_node = self.steer(nearest_node, rnd_node, self.step_size)
        if new_node is None:
            return

        if is_collision_free(nearest_node.state, new_node.state, self.obstacles):
            near_inds = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_inds)

            if new_node:
                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                # Try to connect to goal
                if self.calc_dist_to_goal(new_node.state) <= self.goal_connect_radius:
                    if is_collision_free(new_node.state, self.goal.state, self.obstacles):
                        goal_node = Node(self.goal.state)
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + self.calc_dist_to_goal(new_node.state)
                        self.node_list.append(goal_node)

                        if goal_node.cost < self.best_cost:
                            self.best_cost = goal_node.cost
                            self.best_goal_node = goal_node

    def get_best_solution(self):
        """Return (path, success, best_cost_estimate)."""
        path = self.get_best_path(goal_tolerance=self.goal_tolerance)
        if path is None:
            return None, False, float("inf")

        # For vanilla RRT*, objective == geometric length
        cost = float(self.path_length(path))
        return path, True, cost

    # -----------------------
    # Original logic helpers
    # -----------------------
    def path_length(self, path):
        path = np.asarray(path)
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(to_node.state)
        d, _ = self.calc_distance_and_angle(from_node, to_node)

        if d < 1e-6:
            return None

        if extend_length > d:
            extend_length = d

        new_node.state = from_node.state + (to_node.state - from_node.state) / d * extend_length
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node

    def get_random_node(self):
        # Goal biasing
        if self.rng is None:
            if np.random.random() < self.goal_sample_rate:
                return Node(self.goal.state)
            return Node(sample_uniform(self.bounds))
        else:
            if self.rng.random() < self.goal_sample_rate:
                return Node(self.goal.state)
            return Node(sample_uniform(self.bounds, rng=self.rng))

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [np.sum((node.state - rnd_node.state) ** 2) for node in node_list]
        return int(np.argmin(dlist))

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        dim = 3
        gamma = self.search_radius * 2.0
        r = min(gamma * (np.log(nnode) / nnode) ** (1.0 / dim), self.search_radius)
        dlist = [np.sum((node.state - new_node.state) ** 2) for node in self.node_list]
        return [i for i, d in enumerate(dlist) if d <= r ** 2]

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []
        parents = []

        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and is_collision_free(near_node.state, t_node.state, self.obstacles):
                costs.append(near_node.cost + get_dist(near_node.state, t_node.state))
                parents.append(near_node)

        if not costs:
            return None

        k = int(np.argmin(costs))
        best_parent = parents[k]
        new_node = self.steer(best_parent, new_node)
        new_node.parent = best_parent
        new_node.cost = float(costs[k])
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue

            new_cost = new_node.cost + get_dist(new_node.state, near_node.state)
            if is_collision_free(new_node.state, near_node.state, self.obstacles) and near_node.cost > new_cost:
                old_cost = near_node.cost
                near_node.parent = new_node
                near_node.cost = new_cost
                self.propagate_cost_to_children(near_node, old_cost)

    def propagate_cost_to_children(self, parent_node, old_cost):
        cost_diff = parent_node.cost - old_cost
        for node in self.node_list:
            if node.parent == parent_node:
                prev = node.cost
                node.cost += cost_diff
                self.propagate_cost_to_children(node, prev)

    def calc_distance_and_angle(self, from_node, to_node):
        d = float(np.linalg.norm(to_node.state - from_node.state))
        return d, None

    def calc_dist_to_goal(self, state):
        return float(np.linalg.norm(state - self.goal.state))

    def get_best_path(self, goal_tolerance=2.0):
        candidates = [n for n in self.node_list if self.calc_dist_to_goal(n.state) <= goal_tolerance]
        if not candidates:
            return None

        best = min(candidates, key=lambda n: n.cost)

        path = []
        cur = best
        while cur is not None:
            path.append(cur.state)
            cur = cur.parent
        path = path[::-1]

        # ensure ends at exact goal
        if np.linalg.norm(path[-1] - self.goal.state) > 1e-9:
            path.append(self.goal.state.copy())

        return np.array(path)
