import numpy as np
from planners.rrt_star import RRTStar, Node
from common.utils import sample_uniform, get_dist

class InformedRRTStar(RRTStar):
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter, search_radius)
        self.c_best = float("inf") # Best path cost found so far

    def plan(self, max_time=5.0):
        # Override plan to update c_best and use informed sampling
        import time
        start_time = time.time()
        
        for i in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Informed Sampling
            rnd_node = self.get_informed_state()
            
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.step_size)

            if self.check_collision(new_node, self.obstacles):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
                    
                    # Update best cost if goal is reached/improved
                    if self.calc_dist_to_goal(new_node.state) <= self.step_size: # approximate check
                         # more precise check usually involves rewiring to goal node 
                         # or checking if new_node connects to goal with better cost
                         dist_to_goal = self.calc_dist_to_goal(new_node.state)
                         current_total_cost = new_node.cost + dist_to_goal
                         if current_total_cost < self.c_best:
                             self.c_best = current_total_cost
                             # Note: We don't explicitly add goal node to tree until final extraction 
                             # or if we treat goal as a node. 
                             # For simplicity, c_best is tracked.

        return self.get_best_path()

    def check_collision(self, node, obstacles):
        # We need to check the edge from parent to node
        # In RRTStar.steer, parent is set.
        from common.utils import is_collision_free
        if node.parent:
            return is_collision_free(node.parent.state, node.state, obstacles)
        return True

    def get_informed_state(self):
        if self.c_best == float("inf"):
            return self.get_random_node()
        
        # Ellipsoidal sampling
        c_min = get_dist(self.start.state, self.goal.state)
        x_start = self.start.state
        x_goal = self.goal.state
        x_center = (x_start + x_goal) / 2.0
        
        if self.c_best < c_min:
             # Should not happen theoretically unless cost mismatch
             self.c_best = c_min 

        # Calculate rotation matrix
        # 1. Align x-axis with x_goal - x_start
        a1 = (x_goal - x_start) / c_min
        # Gram-Schmidt for other axes
        # (Simplified 3D rotation alignment)
        # SVD is robust
        U, S, V = np.linalg.svd(np.outer(a1, [1, 0, 0])) # This might be approximate
        # Better:
        # Create Rotation C from body frame to world frame
        # First column is a1 = direction
        # Second and third are orthogonal
        
        # Let's use simple rejection sampling if alignment is complex to implement quickly correctly?
        # Direct ellipsoid sampling is more efficient.
        
        # Construct Rotation Matrix C
        # M = a1 * [1, 0, 0]^T
        # U * diag([1, 1, det(U)det(V)]) * V^T  is rotation?
        # Standard approach:
        # 1. Sample in unit ball
        # 2. Stretch by diag(c_best/2, sqrt(c_best^2 - c_min^2)/2, ...)
        # 3. Rotate and Translate
        
        # Rotation Matrix to align [1,0,0] with (x_goal - x_start)
        # (Using a helper function helps)
        C = self.rotation_matrix_from_vectors(np.array([1, 0, 0]), (x_goal - x_start) / c_min)
        
        L = np.diag([self.c_best / 2.0, 
                     np.sqrt(self.c_best**2 - c_min**2) / 2.0,
                     np.sqrt(self.c_best**2 - c_min**2) / 2.0])
        
        while True:
            xs = self.sample_unit_ball()
            x_rand = np.dot(np.dot(C, L), xs) + x_center
            
            # Check bounds
            if np.all(x_rand >= self.bounds[:, 0]) and np.all(x_rand <= self.bounds[:, 1]):
                return Node(x_rand)
    
    def sample_unit_ball(self):
        while True:
            x = np.random.uniform(-1, 1, 3)
            if np.linalg.norm(x) <= 1:
                return x

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        if any(v): # if not parallel
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        else:
            return np.eye(3) # Parallel or anti-parallel (handling anti-parallel requires specific handling)
            # For this specific case (Start -> Goal), [1,0,0] to direction
            # If anti-parallel, just flip 180 deg around any orthogonal axis.
