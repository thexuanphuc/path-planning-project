# planners/rrt.py
import numpy as np
from common.utils import get_dist, is_collision_free, sample_uniform


class Node:
    def __init__(self, state):
        self.state = np.array(state, dtype=float)
        self.parent = None


class RRT:
    """
    Simple RRT (no rewiring).
    Compatible with the unified experiment harness:
      - step()
      - get_best_solution()
      - uses rng passed from main via constructor
    """

    def __init__(
        self,
        start,
        goal,
        obstacles,
        bounds,
        step_size=0.5,
        max_iter=1000,
        goal_sample_rate=0.2,
        goal_tolerance=2.0,
        rng=None,
        collision_step=0.5,
    ):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds

        self.step_size = float(step_size)
        self.max_iter = int(max_iter)  # kept for parity; wrapper controls stopping
        self.goal_sample_rate = float(goal_sample_rate)
        self.goal_tolerance = float(goal_tolerance)
        self.collision_step = float(collision_step)

        self.rng = rng  # np.random.Generator preferred

        self.node_list = [self.start]
        self.iterations = 0

        # best solution tracking
        self.best_goal_node = None
        self.best_cost = float("inf")

    # -----------------------
    # Incremental interface
    # -----------------------
    def step(self):
        """Run exactly one RRT iteration."""
        self.iterations += 1

        rnd_node = self._get_random_node()
        nearest = self._nearest_node(rnd_node)

        new_node = self._steer(nearest, rnd_node, self.step_size)
        if new_node is None:
            return

        if not is_collision_free(nearest.state, new_node.state, self.obstacles, step_size=self.collision_step):
            return

        self.node_list.append(new_node)

        # goal check
        dist_to_goal = get_dist(new_node.state, self.goal.state)
        if dist_to_goal <= self.goal_tolerance:
            # In simple RRT, treat reaching the goal region as success
            cost = self._cost_to_come(new_node) + dist_to_goal
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_goal_node = new_node

    def get_best_solution(self):
        """
        Returns (path, success, total_cost).
        total_cost for simple RRT = geometric path length to the chosen goal-reaching node (+ final dist-to-goal).
        """
        if self.best_goal_node is None:
            return None, False, float("inf")

        path = self._reconstruct_path(self.best_goal_node)

        # ensure ends exactly at goal for consistent metrics
        if get_dist(path[-1], self.goal.state) > 1e-9:
            path = np.vstack([path, self.goal.state.copy()])

        total_cost = float(self._path_length(path))
        return path, True, total_cost

    # -----------------------
    # Helpers
    # -----------------------
    def _get_random_node(self):
        # goal bias
        r = self.rng.random() if self.rng is not None else np.random.random()
        if r < self.goal_sample_rate:
            return Node(self.goal.state)

        s = sample_uniform(self.bounds, rng=self.rng)
        return Node(s)

    def _nearest_node(self, rnd_node):
        # brute-force nearest neighbor
        d2 = [np.sum((n.state - rnd_node.state) ** 2) for n in self.node_list]
        return self.node_list[int(np.argmin(d2))]

    def _steer(self, from_node, to_node, extend_length):
        d = get_dist(from_node.state, to_node.state)
        if d < 1e-9:
            return None

        L = min(float(extend_length), float(d))
        direction = (to_node.state - from_node.state) / d
        new_state = from_node.state + direction * L

        new_node = Node(new_state)
        new_node.parent = from_node
        return new_node

    def _reconstruct_path(self, node):
        path = []
        cur = node
        while cur is not None:
            path.append(cur.state)
            cur = cur.parent
        return np.array(path[::-1])

    def _path_length(self, path):
        path = np.asarray(path)
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))

    def _cost_to_come(self, node):
        # compute cost by backtracking (simple, fine for class project)
        cost = 0.0
        cur = node
        while cur.parent is not None:
            cost += get_dist(cur.state, cur.parent.state)
            cur = cur.parent
        return float(cost)
