import numpy as np
from planner.rrt_star import RRTStar, Node
from common.utils import sample_uniform, get_dist, is_collision_free

class CustomRRTStar(RRTStar):
    """
    Custom RRT* that reuses planners.rrt_star.RRTStar and overrides only:
      - sampling: goal + direction bias + uniform
      - parent selection: clearance-aware cost
      - rewiring: clearance-aware cost

    This ensures a fair comparison against baseline RRT*.
    """

    def __init__(self, start, goal, obstacles, bounds,
                 step_size=0.5, max_iter=1000, search_radius=1.5,
                 clearance_weight=1.0,
                 p_goal_sample=0.10,
                 p_dir_bias=0.50,
                 dir_bias_power=1.5,
                 clearance_eps=1e-6,
                 rng=None,
                 goal_connect_radius=5.0,
                 goal_tolerance=2.0):
        super().__init__(
            start, goal, obstacles, bounds,
            step_size=step_size, max_iter=max_iter, search_radius=search_radius,
            rng=rng,
            goal_sample_rate=0.0,  # we do our own goal sampling here
            goal_connect_radius=goal_connect_radius,
            goal_tolerance=goal_tolerance
        )

        self.clearance_weight = float(clearance_weight)
        self.p_goal_sample = float(p_goal_sample)
        self.p_dir_bias = float(p_dir_bias)
        self.dir_bias_power = float(dir_bias_power)
        self.clearance_eps = float(clearance_eps)

        self._prev_state = self.start.state.copy()

    # -------- sampling override --------
    def get_random_node(self):
        r = (np.random.random() if self.rng is None else self.rng.random())

        if r < self.p_goal_sample:
            return Node(self.goal.state)

        if r < (self.p_goal_sample + self.p_dir_bias):
            s = self._sample_goal_direction_biased(self._prev_state, self.goal.state)
            return Node(s)

        return Node(sample_uniform(self.bounds, rng=self.rng))

    def _sample_goal_direction_biased(self, prev, goal):
        prev = np.asarray(prev, dtype=float)
        goal = np.asarray(goal, dtype=float)

        x = np.zeros_like(prev)
        for d, (lo, hi) in enumerate(self.bounds):
            lo = float(lo); hi = float(hi)

            if goal[d] >= prev[d]:
                a = float(np.clip(prev[d], lo, hi))
                b = hi
            else:
                a = lo
                b = float(np.clip(prev[d], lo, hi))

            if (b - a) < 1e-9:
                a, b = lo, hi

            u = (np.random.random() if self.rng is None else self.rng.random())
            u_skew = u ** self.dir_bias_power
            x[d] = a + u_skew * (b - a)

        return x

    def step(self):
        # run one base iteration, but when a node is successfully added update prev anchor
        before_n = len(self.node_list)
        super().step()
        after_n = len(self.node_list)
        if after_n > before_n:
            # last added is the newest; update direction anchor
            self._prev_state = self.node_list[-1].state.copy()

    # -------- clearance-aware costs --------
    def get_clearance_cost(self, state):
        state = np.asarray(state, dtype=float)
        min_dist = float("inf")

        for obs in self.obstacles:
            d = float("inf")
            if hasattr(obs, "radius") and hasattr(obs, "center"):
                d = np.linalg.norm(np.asarray(obs.center) - state) - float(obs.radius)
            elif hasattr(obs, "extents") and hasattr(obs, "center"):
                d = np.linalg.norm(np.asarray(obs.center) - state) - float(np.max(obs.extents))
            min_dist = min(min_dist, d)

        if min_dist <= 0.0:
            return 1e6

        return self.clearance_weight * np.exp(-min_dist) + self.clearance_eps

    # -------- parent selection override --------
    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        node_clear_cost = self.get_clearance_cost(new_node.state)
        best_parent = None
        best_cost = float("inf")

        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and is_collision_free(near_node.state, t_node.state, self.obstacles):
                cand = near_node.cost + get_dist(near_node.state, t_node.state) + node_clear_cost
                if cand < best_cost:
                    best_cost = cand
                    best_parent = near_node

        if best_parent is None:
            return None

        new_node = self.steer(best_parent, new_node)
        new_node.parent = best_parent
        new_node.cost = best_cost
        return new_node

    # -------- rewiring override --------
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue

            new_cost = (
                new_node.cost
                + get_dist(new_node.state, near_node.state)
                + self.get_clearance_cost(near_node.state)
            )

            if is_collision_free(new_node.state, near_node.state, self.obstacles) and near_node.cost > new_cost:
                old_cost = near_node.cost
                near_node.parent = new_node
                near_node.cost = new_cost
                self.propagate_cost_to_children(near_node, old_cost)

    def get_best_solution(self):
        path = self.get_best_path(goal_tolerance=self.goal_tolerance)
        if path is None:
            return None, False, float("inf")
        # objective for custom includes clearance penalties in tree-cost,
        # but for standardized output we store:
        # - path_length = geometry length
        # - total_cost = best node cost (your objective)
        geom = float(self.path_length(path))
        # best node cost (objective)
        candidates = [n for n in self.node_list if self.calc_dist_to_goal(n.state) <= self.goal_tolerance]
        total_cost = float(min(candidates, key=lambda n: n.cost).cost) if candidates else geom
        return path, True, total_cost
