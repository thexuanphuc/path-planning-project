import numpy as np
import time
from common.utils import get_dist, is_collision_free, sample_uniform

class Node:
    def __init__(self, state):
        self.state = np.array(state)
        self.parent = None
        self.g_score = np.inf
        self.f_score = np.inf
        self.children = []

    def __repr__(self):
        return f"Node(state={self.state}, g={self.g_score}, f={self.f_score})"

class BITStar:
    def __init__(self, start, goal, obstacles, bounds, eta=1.1, batch_size=100):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.eta = eta # Inflation factor for radius
        self.batch_size = batch_size

        self.start_node = Node(self.start)
        self.goal_node = Node(self.goal)
        
        self.V = [] # Vertex set (tree)
        self.X_samples = [] # Unconnected samples
        self.QE = [] # Edge queue
        self.QV = [] # Vertex queue
        
        self.r = np.inf # Current connection radius
        self.g_goal = np.inf # Cost to goal

    def get_heuristic(self, state):
        return np.linalg.norm(state - self.goal)

    def plan(self, max_time=5.0):
        start_time = time.time()
        
        # Initialize
        self.start_node.g_score = 0
        self.start_node.f_score = self.get_heuristic(self.start)
        self.V.append(self.start_node)
        self.X_samples = [] # Initially empty, will be filled in batches
        
        # Main loop
        while time.time() - start_time < max_time:
            if not self.QE and not self.QV:
                self.prune()
                self.X_samples.extend(self.sample_batch(self.batch_size))
                self.V.extend([node for node in self.QV]) # Re-add vertices if they were put back? No, QV is for expansion
                
                # Update radius
                # Simple radius approach for RRT* k-nearest or r-disc. 
                # For BIT*, radius decreases as samples increase.
                # r = eta * 2 * (1 + 1/d)^(1/d) * (measure(X_free)/zeta_d)^(1/d) * (log(q)/q)^(1/d)
                # Simplified: depends on number of samples q
                q = len(self.V) + len(self.X_samples)
                dim = len(self.start)
                self.r = self.eta * 5.0 * (np.log(q) / q) ** (1/dim) # Tunable 5.0 factor
                
                # Make existing vertices candidates for expansion
                self.QV = [v for v in self.V if v.f_score < self.g_goal]
                
            # Expand best vertex
            while self.QV:
                # Sort QV by g_score + heuristic estimate to sample? 
                # BIT* usually expands consistently.
                # Here we just iterate through all in QV to find potential edges
                v_curr = self.QV.pop(0) # Simplification: Treat QV as list of nodes to expand
                
                # Find neighbors in X_samples within radius r
                # In robust impl, use KDTree. Here brute force for simplicity since N is small-ish
                for x_state in self.X_samples:
                    dist = np.linalg.norm(v_curr.state - x_state)
                    if dist <= self.r:
                        # Improved heuristic check
                        g_estimated = v_curr.g_score + dist + self.get_heuristic(x_state)
                        if g_estimated < self.g_goal:
                            self.QE.append((v_curr, x_state, dist)) # Edge candidate
            
            # Sort QE by total estimated cost: g(v) + dist + h(x)
            self.QE.sort(key=lambda edge: edge[0].g_score + edge[2] + self.get_heuristic(edge[1]))

            # Expand best edge
            if self.QE:
                v, x_state, dist = self.QE.pop(0) # Best edge
                
                # Check actual collision
                if is_collision_free(v.state, x_state, self.obstacles):
                    # check if x is already in V (rewiring?)
                    # Simplified: Assume x is new sample for now.
                    # In full BIT*, X_samples are states, not nodes yet.
                    
                    # Create new node
                    new_node = Node(x_state)
                    new_node.parent = v
                    new_node.g_score = v.g_score + dist
                    new_node.f_score = new_node.g_score + self.get_heuristic(x_state)
                    
                    # If x_state was in X_samples, remove it
                    # (Note: manual list removal is slow, valid for demo)
                    # self.X_samples.remove(x_state) # - this might be tricky with numpy arrays comparison
                    # Better to handle index or just mark used.
                    
                    # Check if goal reached
                    if np.linalg.norm(new_node.state - self.goal) <= 1.0: # Goal tolerance
                        if new_node.g_score < self.g_goal:
                            self.g_goal = new_node.g_score
                            self.goal_node = new_node
                            print(f"Goal reached! Cost: {self.g_goal:.4f}")

                    self.V.append(new_node)
                    self.QV.append(new_node)
                    
                    # Remove from samples to avoid re-expanding same state same way
                    # (Actually BIT* keeps it for rewiring, but simplified version: just added to tree)
                    # For strict BIT*, we need to manage V and X_samples sets carefully.
                    # Let's try to remove it from X_samples to prevent duplicates in V
                    # In efficient impl, use set or KDTree.
                    # Finding the array match:
                    for i, s in enumerate(self.X_samples):
                        if np.array_equal(s, x_state):
                            self.X_samples.pop(i)
                            break

        return self.reconstruct_path()

    def sample_batch(self, k):
        # Informed sampling could be here
        samples = []
        for _ in range(k):
             samples.append(sample_uniform(self.bounds))
        return samples

    def prune(self):
        # Remove nodes with f_score > g_goal
        self.V = [v for v in self.V if v.f_score < self.g_goal]
        self.X_samples = [x for x in self.X_samples if self.get_heuristic(x) < self.g_goal]

    def reconstruct_path(self):
        if self.g_goal == np.inf:
            return None
        
        path = []
        curr = self.goal_node
        while curr:
            path.append(curr.state)
            curr = curr.parent
        return path[::-1]

