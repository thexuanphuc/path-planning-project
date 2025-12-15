import numpy as np
import time
from common.utils import get_dist, is_collision_free, sample_uniform

class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = [self.start]

    def plan(self, max_time=5.0):
        start_time = time.time()
        for i in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.step_size)
            
            if new_node is None:
                continue

            if is_collision_free(nearest_node.state, new_node.state, self.obstacles):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
                    
                    # Try to connect to goal
                    if self.calc_dist_to_goal(new_node.state) <= 5.0:
                        if is_collision_free(new_node.state, self.goal.state, self.obstacles):
                            goal_node = Node(self.goal.state)
                            goal_node.parent = new_node
                            goal_node.cost = new_node.cost + self.calc_dist_to_goal(new_node.state)
                            self.node_list.append(goal_node)
                            print(f"Goal connected! Cost: {goal_node.cost:.2f}")

        return self.get_best_path()

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(to_node.state)
        d, _ = self.calc_distance_and_angle(from_node, to_node)

        if d < 1e-6:  # Nodes are at same location
            return None

        if extend_length > d:
            extend_length = d

        # Interpolate
        new_node.state = from_node.state + (to_node.state - from_node.state) / d * extend_length
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node

    def get_random_node(self):
        # Goal biasing: 20% chance to sample goal
        if np.random.random() < 0.2:
            return Node(self.goal.state)
        return Node(sample_uniform(self.bounds))

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.state[0] - rnd_node.state[0])**2 + (node.state[1] - rnd_node.state[1])**2 + (node.state[2] - rnd_node.state[2])**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.search_radius * np.sqrt((np.log(nnode) / nnode))
        r = min(r, self.search_radius) 
        # For simplicity in 3D, fixed radius or k-nearest often used. 
        # Using fixed search_radius for now to match arguments, or dynamic if preferred.
        # Let's use the provided fixed search radius as a base or cap.
        
        dlist = [(node.state[0] - new_node.state[0])**2 + (node.state[1] - new_node.state[1])**2 + (node.state[2] - new_node.state[2])**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r ** 2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None
        
        costs = []
        possible_parents = []

        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and is_collision_free(near_node.state, t_node.state, self.obstacles):
                costs.append(near_node.cost + get_dist(near_node.state, t_node.state))
                possible_parents.append(near_node)
        
        if not costs:
            return None

        min_cost = min(costs)
        min_ind = costs.index(min_cost)
        new_node = self.steer(possible_parents[min_ind], new_node) # Re-steer to set correct parent/cost
        new_node.parent = possible_parents[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue

            edge_node.cost = new_node.cost + get_dist(new_node.state, near_node.state)
            
            if is_collision_free(new_node.state, near_node.state, self.obstacles) and near_node.cost > edge_node.cost:
                near_node.parent = new_node
                near_node.cost = edge_node.cost
                # Propagate cost update? RRT* usually does. 
                # For basic implementation, we skip full tree propagation for speed/simplicity,
                # but it affects optimality.

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.state[0] - from_node.state[0]
        dy = to_node.state[1] - from_node.state[1]
        dz = to_node.state[2] - from_node.state[2]
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        return d, None

    def calc_dist_to_goal(self, state):
        return np.linalg.norm(state - self.goal.state)

    def get_best_path(self):
        # find node closest to goal or within tolerance
        best_node = None
        min_dist = float("inf")
        
        # Check tolerance
        tolerance = 2.0 # Goal region
        
        candidate_nodes = []
        for node in self.node_list:
            dist = self.calc_dist_to_goal(node.state)
            if dist <= tolerance:
                # Check collision free to goal if needed, or assume tolerance is goal region
                candidate_nodes.append(node)

        if not candidate_nodes:
            return None

        # Find min cost among candidates
        best_node = min(candidate_nodes, key=lambda n: n.cost)
        
        path = []
        curr = best_node
        while curr:
            path.append(curr.state)
            curr = curr.parent
        # Add goal if not exactly at goal?
        path.insert(0, self.goal.state)
        return np.array(path[::-1])
