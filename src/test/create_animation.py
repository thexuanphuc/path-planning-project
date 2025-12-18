import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from common.utils import Sphere, is_collision_free, sample_uniform, get_dist
from planner.rrt_star import Node

class AnimatedRRTStar:
    """RRT* variant that records tree growth for animation"""
    def __init__(self, start, goal, obstacles, bounds, step_size=0.8, max_iter=500, search_radius=3.0):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = [self.start]
        
        # Animation tracking
        self.snapshots = []  # List of (nodes, edges, goal_found) tuples
        self.goal_found = False
        
    def record_snapshot(self):
        """Record current tree state"""
        nodes = [n.state.copy() for n in self.node_list]
        edges = [(n.parent.state.copy(), n.state.copy()) for n in self.node_list if n.parent]
        self.snapshots.append((nodes, edges, self.goal_found))
    
    def plan(self, max_time=10.0, snapshot_interval=10):
        start_time = time.time()
        self.record_snapshot()  # Initial state
        
        for i in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Goal biasing
            if np.random.random() < 0.2:
                rnd_node = Node(self.goal.state)
            else:
                rnd_node = Node(sample_uniform(self.bounds))
            
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
                            self.goal_found = True
                    
                    # Record snapshot periodically
                    if i % snapshot_interval == 0 or self.goal_found:
                        self.record_snapshot()
        
        # Final snapshot
        self.record_snapshot()
        return self.get_best_path()
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(to_node.state)
        d = np.linalg.norm(to_node.state - from_node.state)
        
        if d < 1e-6:
            return None
        
        if extend_length > d:
            extend_length = d
        
        new_node.state = from_node.state + (to_node.state - from_node.state) / d * extend_length
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node
    
    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [np.sum((node.state - rnd_node.state)**2) for node in node_list]
        return dlist.index(min(dlist))
    
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = min(self.search_radius * np.sqrt(np.log(nnode) / nnode), self.search_radius)
        dlist = [np.sum((node.state - new_node.state)**2) for node in self.node_list]
        return [i for i, d in enumerate(dlist) if d <= r ** 2]
    
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
        new_node = self.steer(possible_parents[min_ind], new_node)
        new_node.parent = possible_parents[min_ind]
        new_node.cost = min_cost
        return new_node
    
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            
            edge_node.cost = new_node.cost + get_dist(new_node.state, near_node.state)
            
            if is_collision_free(new_node.state, near_node.state, self.obstacles) and near_node.cost > edge_node.cost:
                near_node.parent = new_node
                near_node.cost = edge_node.cost
    
    def calc_dist_to_goal(self, state):
        return np.linalg.norm(state - self.goal.state)
    
    def get_best_path(self):
        tolerance = 2.0
        candidate_nodes = [node for node in self.node_list if self.calc_dist_to_goal(node.state) <= tolerance]
        
        if not candidate_nodes:
            return None
        
        best_node = min(candidate_nodes, key=lambda n: n.cost)
        path = []
        curr = best_node
        while curr:
            path.append(curr.state)
            curr = curr.parent
        path.insert(0, self.goal.state)
        return np.array(path[::-1])


def create_animation():
    # Environment
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    print("Running RRT* and recording snapshots...")
    planner = AnimatedRRTStar(start, goal, obstacles, bounds, step_size=0.8, max_iter=300, search_radius=3.0)
    path = planner.plan(max_time=10.0, snapshot_interval=5)
    
    print(f"Recorded {len(planner.snapshots)} snapshots")
    if path is not None:
        print(f"Path found with cost: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}")
    
    # Create animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_zlim(bounds[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Draw obstacles
        for obs in obstacles:
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
            x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
            y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
            z = obs.center[2] + obs.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)
        
        # Draw start and goal
        ax.scatter(start[0], start[1], start[2], c='green', marker='o', s=150, label='Start', zorder=10)
        ax.scatter(goal[0], goal[1], goal[2], c='red', marker='*', s=150, label='Goal', zorder=10)
        
        # Get current snapshot
        nodes, edges, goal_found = planner.snapshots[frame]
        
        # Draw tree edges
        for parent_state, child_state in edges:
            ax.plot([parent_state[0], child_state[0]], 
                   [parent_state[1], child_state[1]], 
                   [parent_state[2], child_state[2]], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # Draw nodes
        if len(nodes) > 0:
            nodes_array = np.array(nodes)
            ax.scatter(nodes_array[:, 0], nodes_array[:, 1], nodes_array[:, 2], 
                      c='blue', s=10, alpha=0.6)
        
        # Draw final path if goal found
        if goal_found and path is not None:
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   'g-', linewidth=3, label='Path', zorder=5)
        
        status = "Goal Found!" if goal_found else "Exploring..."
        ax.set_title(f'RRT* Progress - Frame {frame+1}/{len(planner.snapshots)} - Nodes: {len(nodes)} - {status}')
        ax.legend()
        ax.view_init(elev=25, azim=45 + frame * 2)  # Slowly rotate view
    
    print("Creating animation...")
    anim = FuncAnimation(fig, update, frames=len(planner.snapshots), interval=200, repeat=True)
    
    print("Saving GIF (this may take a minute)...")
    writer = PillowWriter(fps=5)
    anim.save('rrt_star_animation.gif', writer=writer, dpi=80)
    print("Animation saved to rrt_star_animation.gif")
    plt.close()

if __name__ == "__main__":
    create_animation()
