import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import time

from common.utils import Sphere, is_collision_free, sample_uniform, get_dist
from planner.rrt_star import Node

class AnimatedPlanner:
    """Base planner that records tree growth for animation"""
    def __init__(self, name, start, goal, obstacles, bounds, color='blue'):
        self.name = name
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.node_list = [self.start]
        self.color = color
        
        # Animation tracking
        self.snapshots = []
        self.goal_found = False
        self.final_path = None
        
    def record_snapshot(self):
        """Record current tree state"""
        nodes = [n.state.copy() for n in self.node_list]
        edges = [(n.parent.state.copy(), n.state.copy()) for n in self.node_list if n.parent]
        self.snapshots.append((nodes, edges, self.goal_found))


class AnimatedRRTStar(AnimatedPlanner):
    def __init__(self, name, start, goal, obstacles, bounds, color='blue', step_size=0.8, max_iter=200, search_radius=3.0):
        super().__init__(name, start, goal, obstacles, bounds, color)
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
    
    def plan(self, max_time=8.0, snapshot_interval=5):
        start_time = time.time()
        self.record_snapshot()
        
        for i in range(self.max_iter):
            if time.time() - start_time > max_time:
                break
            
            # Goal biasing
            if np.random.random() < 0.2:
                rnd_node = Node(self.goal.state)
            else:
                rnd_node = Node(sample_uniform(self.bounds))
            
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node)
            
            if new_node is None:
                continue
            
            if is_collision_free(nearest_node.state, new_node.state, self.obstacles):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)
                    
                    if np.linalg.norm(new_node.state - self.goal.state) <= 5.0:
                        if is_collision_free(new_node.state, self.goal.state, self.obstacles):
                            goal_node = Node(self.goal.state)
                            goal_node.parent = new_node
                            goal_node.cost = new_node.cost + np.linalg.norm(new_node.state - self.goal.state)
                            self.node_list.append(goal_node)
                            self.goal_found = True
                    
                    if i % snapshot_interval == 0 or self.goal_found:
                        self.record_snapshot()
        
        self.record_snapshot()
        self.final_path = self.get_best_path()
        return self.final_path
    
    def steer(self, from_node, to_node):
        d = np.linalg.norm(to_node.state - from_node.state)
        if d < 1e-6:
            return None
        extend_length = min(self.step_size, d)
        new_node = Node(from_node.state + (to_node.state - from_node.state) / d * extend_length)
        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node
    
    def get_nearest_node_index(self, rnd_node):
        dlist = [np.sum((node.state - rnd_node.state)**2) for node in self.node_list]
        return dlist.index(min(dlist))
    
    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = min(self.search_radius * np.sqrt(np.log(nnode) / nnode), self.search_radius)
        dlist = [np.sum((node.state - new_node.state)**2) for node in self.node_list]
        return [i for i, d in enumerate(dlist) if d <= r ** 2]
    
    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None
        costs, possible_parents = [], []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and is_collision_free(near_node.state, t_node.state, self.obstacles):
                costs.append(near_node.cost + get_dist(near_node.state, t_node.state))
                possible_parents.append(near_node)
        if not costs:
            return None
        min_ind = costs.index(min(costs))
        new_node = self.steer(possible_parents[min_ind], new_node)
        new_node.parent = possible_parents[min_ind]
        new_node.cost = costs[min_ind]
        return new_node
    
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_cost = new_node.cost + get_dist(new_node.state, near_node.state)
            if is_collision_free(new_node.state, near_node.state, self.obstacles) and near_node.cost > edge_cost:
                near_node.parent = new_node
                near_node.cost = edge_cost
    
    def get_best_path(self):
        candidates = [n for n in self.node_list if np.linalg.norm(n.state - self.goal.state) <= 2.0]
        if not candidates:
            return None
        best = min(candidates, key=lambda n: n.cost)
        path = []
        curr = best
        while curr:
            path.append(curr.state)
            curr = curr.parent
        path.insert(0, self.goal.state)
        return np.array(path[::-1])


def draw_subplot(ax, bounds, start, goal, obstacles, planner, frame_idx):
    """Draw a single planner's state"""
    ax.clear()
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.tick_params(labelsize=6)
    
    # Draw obstacles
    for obs in obstacles:
        u, v = np.mgrid[0:2*np.pi:10j, 0:np.pi:6j]
        x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
        y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
        z = obs.center[2] + obs.radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)
    
    # Draw start and goal
    ax.scatter(start[0], start[1], start[2], c='green', marker='o', s=80, zorder=10)
    ax.scatter(goal[0], goal[1], goal[2], c='red', marker='*', s=80, zorder=10)
    
    # Get snapshot for this frame
    if frame_idx < len(planner.snapshots):
        nodes, edges, goal_found = planner.snapshots[frame_idx]
        
        # Draw tree edges
        for parent_state, child_state in edges:
            ax.plot([parent_state[0], child_state[0]], 
                   [parent_state[1], child_state[1]], 
                   [parent_state[2], child_state[2]], 
                   color=planner.color, alpha=0.2, linewidth=0.5)
        
        # Draw nodes
        if len(nodes) > 0:
            nodes_array = np.array(nodes)
            ax.scatter(nodes_array[:, 0], nodes_array[:, 1], nodes_array[:, 2], 
                      c=planner.color, s=5, alpha=0.4)
        
        # Draw final path if found
        if goal_found and planner.final_path is not None:
            path = planner.final_path
            ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                   color=planner.color, linewidth=2.5, zorder=5)
        
        status = "✓ Goal!" if goal_found else f"Nodes: {len(nodes)}"
        ax.set_title(f'{planner.name}\n{status}', fontsize=10, fontweight='bold')
    
    # Rotate camera view based on frame
    azim = 45 + (frame_idx * 3)  # Rotate 3 degrees per frame
    ax.view_init(elev=25, azim=azim)


def create_comparison_animation():
    # Environment
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    print("Running planners and recording snapshots...")
    
    # Create all planners
    planners = [
        AnimatedRRTStar("RRT*", start, goal, obstacles, bounds, color='red', 
                       step_size=0.8, max_iter=200, search_radius=3.0),
        AnimatedRRTStar("RRT* (Run 2)", start, goal, obstacles, bounds, color='blue', 
                       step_size=0.8, max_iter=200, search_radius=3.0),
        AnimatedRRTStar("RRT* (Run 3)", start, goal, obstacles, bounds, color='green', 
                       step_size=0.8, max_iter=200, search_radius=3.0),
        AnimatedRRTStar("RRT* (Run 4)", start, goal, obstacles, bounds, color='orange', 
                       step_size=0.8, max_iter=200, search_radius=3.0),
    ]
    
    # Run all planners
    for planner in planners:
        print(f"Running {planner.name}...")
        planner.plan(max_time=8.0, snapshot_interval=4)
        print(f"  {len(planner.snapshots)} snapshots, Goal: {planner.goal_found}")
    
    # Find max snapshots for animation length
    max_snapshots = max(len(p.snapshots) for p in planners)
    print(f"\nCreating animation with {max_snapshots} frames...")
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(14, 12))
    axes = [
        fig.add_subplot(2, 2, 1, projection='3d'),
        fig.add_subplot(2, 2, 2, projection='3d'),
        fig.add_subplot(2, 2, 3, projection='3d'),
        fig.add_subplot(2, 2, 4, projection='3d')
    ]
    
    def update(frame):
        for ax, planner in zip(axes, planners):
            draw_subplot(ax, bounds, start, goal, obstacles, planner, frame)
        fig.suptitle(f'Sampling-Based Planners Comparison - Frame {frame+1}/{max_snapshots}', 
                    fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, update, frames=max_snapshots, interval=300, repeat=True)
    
    print("Saving GIF (this may take 1-2 minutes)...")
    writer = PillowWriter(fps=4)
    anim.save('planners_comparison_animation.gif', writer=writer, dpi=100)
    print("Animation saved to planners_comparison_animation.gif")
    plt.close()

if __name__ == "__main__":
    create_comparison_animation()
