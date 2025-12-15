import numpy as np
from planners.rrt_star import RRTStar, Node
from common.utils import get_dist, sample_uniform

class CustomRRTStar(RRTStar):
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5, clearance_weight=1.0):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter, search_radius)
        self.clearance_weight = clearance_weight

    def get_clearance_cost(self, state):
        # Calculate minimum distance to any obstacle
        min_dist = float("inf")
        for obs in self.obstacles:
            # Approximate distance to obstacle surface
            # For Sphere: dist(center, state) - radius
            # For Box: simple approx or bounding sphere
            d = float("inf")
            if hasattr(obs, 'radius'): # Sphere
                d = np.linalg.norm(obs.center - state) - obs.radius
            elif hasattr(obs, 'extents'): # Box
                 # simplified: distance to center minus max extent? 
                 # Better: 0 if inside (collision checked elsewhere), distance to box otherwise.
                 # Let's use distance to center minus max extent as rough heuristic for 'clearance'
                 # (This is just a heuristic cost, collision is strict)
                 # or just use distance to center
                 d = np.linalg.norm(obs.center - state) - np.max(obs.extents)
            
            if d < min_dist:
                min_dist = d
        
        # We want to maximize clearance -> minimize cost
        # Cost = 1 / (distance + epsilon) or similar
        # Or simply penalize proximity: cost += weight * exp(-distance)
        if min_dist <= 0: return 100.0 # Should be collision
        return self.clearance_weight * np.exp(-min_dist)

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None
        
        costs = []
        possible_parents = []
        
        # Clearance cost for the new node state
        node_clearance_cost = self.get_clearance_cost(new_node.state)

        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacles):
                # New Cost = Parent Cost + Euclidian Dist + Clearance Cost of new state
                # (Clearance is integrated or state cost)
                # Here adding state cost
                costs.append(near_node.cost + get_dist(near_node.state, t_node.state) + node_clearance_cost)
                possible_parents.append(near_node)
        
        if not costs:
            return None

        min_cost = min(costs)
        min_ind = costs.index(min_cost)
        
        new_node = self.steer(possible_parents[min_ind], new_node)
        new_node.parent = possible_parents[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        node_clearance_cost = self.get_clearance_cost(new_node.state)
        
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            
            # Recalculate cost with new parent
            # Assuming near_node also has its own clearance cost intrinsic to it that doesn't change,
            # but the path cost changes.
            # Wait, cost logic: g(n) = g(parent) + dist + clearance(n)
            # So if we change parent, only g(parent) + dist changes.
            
            edge_cost = new_node.cost + get_dist(new_node.state, near_node.state) # + near_node_clearance?
            # We need to know near_node's clearance cost to compare fairly?
            # Or just store total cost.
            # If we strictly follow additive cost:
            # near_node.cost should be updated to edge_cost + get_clearance_cost(near_node.state)
            # if we didn't include it in edge_cost above.
            
            # Let's simplify: rewire if path is shorter Euclidean-wise? 
            # OR rewire if total cost (including clearance) is lower.
            # Since clearance cost of near_node is constant regardless of parent,
            # minimizing (g(parent) + dist) is equivalent to minimizing total cost.
            
            if self.check_collision(edge_node, self.obstacles):
                 # Compare path components
                 # Current cost of near_node: near_node.cost
                 # Proposed cost: new_node.cost + dist(new, near) + (potentially clearance of near if we recalc)
                 
                 # Note: in choose_parent we added clearance cost.
                 # So near_node.cost ALREADY includes its clearance cost.
                 # We should compare:
                 # new_cost = new_node.cost + dist
                 # This 'new_cost' would be the cost ARRIVING at near_node.
                 # But we need to add near_node's clearance to be consistent with how it was created?
                 # Yes.
                 
                 # To properly check, we'd need to separate G (path) vs C (state) costs, 
                 # but for now let's just assume we want to minimize total accumulated cost.
                 
                 # This implementation of rewire might be slightly approximate if cost definitions vary,
                 # but standard RRT* rewire logic:
                 accumulated_cost_via_new = new_node.cost + get_dist(new_node.state, near_node.state) + self.get_clearance_cost(near_node.state)
                 
                 if near_node.cost > accumulated_cost_via_new:
                     near_node.parent = new_node
                     near_node.cost = accumulated_cost_via_new
                     # Ideally propagate to children
    
    def check_collision(self, node, obstacles):
        from common.utils import is_collision_free
        if node.parent:
             return is_collision_free(node.parent.state, node.state, obstacles)
        return True
