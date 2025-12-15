import numpy as np
from planners.rrt_star import RRTStar, Node
from common.utils import sample_uniform, get_dist, is_collision_free
import time

class InformedRRTStar(RRTStar):
    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter, search_radius)
        self.c_best = float("inf") # Best path cost found so far
        self.ellipsoid_coords = [] # Stores (center, radii, C matrix) for visualization

    def plan(self, max_time=5.0):
        start_time = time.time()
        
        for i in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Informed Sampling: Overrides RRTStar.get_random_node() logic
            rnd_node = self.get_informed_state()
            
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
                    
                    # Update best cost (c_best) check
                    dist_to_goal = self.calc_dist_to_goal(new_node.state)
                    current_total_cost = new_node.cost + dist_to_goal
                    
                    if current_total_cost < self.c_best:
                        self.c_best = current_total_cost
                        
                        # --- OUTPUT/STORE ELLIPSOID COORDINATES ---
                        # Only store parameters when c_best is updated
                        center, L_diag, C = self._calculate_ellipsoid_params()
                        self.ellipsoid_coords.append({
                            'c_best': self.c_best,
                            'center': center.copy(),
                            'radii': L_diag.copy(),
                            'rotation': C.copy()
                        })
                        # --- END STORE ---

        return self.get_best_path()

    def _calculate_ellipsoid_params(self):
        """
        Helper to compute the ellipsoid parameters (center, radii, rotation).
        This method implements the core Section IV of the Informed RRT* paper.
        """
        c_min = get_dist(self.start.state, self.goal.state)
        x_start = self.start.state
        x_goal = self.goal.state
        x_center = (x_start + x_goal) / 2.0
        
        c_use = max(self.c_best, c_min * 1.01) # Use c_best, ensuring it's slightly > c_min for calculation
        
        # 1. Radii (Diagonal of L matrix)
        r1 = c_use / 2.0 # Major semi-axis length
        
        # Minor semi-axis length (using sqrt(c_best^2 - c_min^2) / 2.0)
        r_minor = np.sqrt(c_use**2 - c_min**2) / 2.0
        L_diag = np.array([r1, r_minor, r_minor])

        # 2. Rotation Matrix (C) - Aligns x-axis to the fwd vector
        # (Goal - Start) vector is the major axis direction
        fwd_vec = (x_goal - x_start) / c_min 
        e1 = np.array([1.0, 0.0, 0.0]) # The axis we want to rotate
        
        # Use SVD to find the rotation C that maps e1 to fwd_vec
        # This is the standard method for optimal alignment (Wahba's problem solution)
        U, S, Vh = np.linalg.svd(np.outer(fwd_vec, e1))
        
        M = np.eye(3)
        M[2, 2] = np.linalg.det(U) * np.linalg.det(Vh) # Vh is V transpose, so det(Vh)=det(V)
        C = U @ M @ Vh
        
        return x_center, L_diag, C
        
    def get_informed_state(self):
        """Overrides RRTStar's sampling to draw samples from the informed subset (ellipsoid)."""
        if self.c_best == float("inf"):
            # Fall back to uniform sampling if no solution found yet
            return self.get_random_node()
        
        x_center, L_diag, C = self._calculate_ellipsoid_params()
        L = np.diag(L_diag)
        
        # Sample uniformly from the ellipsoid
        while True:
            xs = self.sample_unit_ball() # Sample point in unit ball
            
            # Transformation: x_rand = C * L * xs + x_center
            x_rand = np.dot(np.dot(C, L), xs) + x_center
            
            # Check bounds (must be within the search space)
            if np.all(x_rand >= self.bounds[:, 0]) and np.all(x_rand <= self.bounds[:, 1]):
                return Node(x_rand)
    
    def sample_unit_ball(self):
        """Samples a random point uniformly from a 3D unit ball."""
        while True:
            # Generate points in a cube [-1, 1]^3
            x = np.random.uniform(-1, 1, 3)
            # Rejection sampling: accept if inside the unit sphere
            if np.linalg.norm(x) <= 1:
                return x
            
def plot_ellipsoid_3d(ellipsoid_params, ax, color='blue', alpha=0.1):
    """
    Plots a 3D ellipsoid given its parameters (center, radii, rotation matrix).
    """
    center = ellipsoid_params['center']
    radii = ellipsoid_params['radii']
    C = ellipsoid_params['rotation']
    
    # Generate points on a unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    points_sphere = np.array([x.flatten(), y.flatten(), z.flatten()])
    
    # Scale, Rotate, and Translate
    points_scaled = points_sphere * radii[:, np.newaxis]
    points_rotated = C @ points_scaled
    points_ellipsoid = points_rotated + center[:, np.newaxis]

    # Reshape back for plotting
    X_e = points_ellipsoid[0, :].reshape(x.shape)
    Y_e = points_ellipsoid[1, :].reshape(y.shape)
    Z_e = points_ellipsoid[2, :].reshape(z.shape)

    ax.plot_surface(X_e, Y_e, Z_e, rstride=3, cstride=3, color=color, alpha=alpha, linewidth=0)
    
    # Optional: Draw the major axis line for clarity
    major_axis_vec = C[:, 0] * radii[0] 
    ax.plot([center[0] - major_axis_vec[0], center[0] + major_axis_vec[0]],
            [center[1] - major_axis_vec[1], center[1] + major_axis_vec[1]],
            [center[2] - major_axis_vec[2], center[2] + major_axis_vec[2]],
            color=color, linestyle='--', linewidth=2, alpha=0.5)

# # --- Example Usage (Demonstration of how to use the class and plot) ---

# if __name__ == '__main__':
#     # Define problem
#     START = np.array([-5.0, -5.0, -5.0])
#     GOAL = np.array([5.0, 5.0, 5.0])
#     OBSTACLES = []
#     BOUNDS = np.array([[-10, 10], [-10, 10], [-10, 10]])

#     # Initialize and run planner
#     planner = InformedRRTStar(
#         start=START,
#         goal=GOAL,
#         obstacles=OBSTACLES,
#         bounds=BOUNDS,
#         step_size=1.0,
#         max_iter=1000
#     )
    
#     print(f"Starting Informed RRT* planning ({planner.max_iter} iterations)...")
#     path = planner.plan(max_time=5.0)
    
#     num_ellipsoids = len(planner.ellipsoid_coords)
#     print(f"\nPlanning finished. Final c_best: {planner.c_best:.2f}")
#     print(f"Number of ellipsoid updates recorded: {num_ellipsoids}")

#     # --- Plotting the Ellipsoid Shrinkage ---
#     if planner.ellipsoid_coords:
#         fig = plt.figure(figsize=(12, 10))
#         ax = fig.add_subplot(111, projection='3d')
        
#         # Set bounds for the plot
#         ax.set_xlim(BOUNDS[0])
#         ax.set_ylim(BOUNDS[1])
#         ax.set_zlim(BOUNDS[2])
#         ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
#         ax.set_title('Informed RRT* Ellipsoid Shrinkage (Evolution of Search Space)')
        
#         # Plot Start/Goal
#         ax.scatter(START[0], START[1], START[2], c='g', marker='o', s=150, label='Start')
#         ax.scatter(GOAL[0], GOAL[1], GOAL[2], c='r', marker='x', s=150, label='Goal')
#         ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]], [START[2], GOAL[2]], 'k:', alpha=0.5)

#         # Plot the final path (if found)
#         if path is not None:
#              ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g-', linewidth=3, label='Found Path')
        
#         # Plot a subset of the recorded ellipsoids (e.g., first 5, middle 5, last 5)
#         indices_to_plot = sorted(list(set([0, num_ellipsoids-1] + [int(i * (num_ellipsoids-1)/5) for i in range(1, 5)])))
#         cmap = plt.cm.viridis
        
#         for i, idx in enumerate(indices_to_plot):
#             if idx < 0 or idx >= num_ellipsoids: continue
#             params = planner.ellipsoid_coords[idx]
            
#             plot_ellipsoid_3d(
#                 params, 
#                 ax=ax, 
#                 color=cmap(i / len(indices_to_plot)), 
#                 alpha=0.2 + 0.5 * (i / len(indices_to_plot))
#             )

#         # Plot the sampled nodes (optional)
#         node_states = np.array([n.state for n in planner.node_list])
#         ax.scatter(node_states[:, 0], node_states[:, 1], node_states[:, 2], c='gray', s=5, alpha=0.3, label='RRT* Nodes')

#         plt.legend()
#         plt.show()
#     elif path is not None:
#         print("Path found, but c_best was never improved from initial state.")
#     else:
#         print("No path found and no ellipsoid updates recorded.")