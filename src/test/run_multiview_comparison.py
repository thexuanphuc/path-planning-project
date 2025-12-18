import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.utils import Sphere, Box
from planner.bit_star import BITStar
from planner.rrt_star import RRTStar
from planner.informed_rrt_star import InformedRRTStar, plot_ellipsoid_3d
from planner.custom_rrt_star import CustomRRTStar

def run_planner(planner_class, name, start, goal, obstacles, bounds, **kwargs):
    print(f"--- Running {name} ---")
    planner = planner_class(start, goal, obstacles, bounds, **kwargs)

    path = planner.plan(max_time=15.0)
        
    cost = float('inf')
    success = False
    
    if path is not None:
        path = np.array(path)
        cost = 0
        for i in range(len(path)-1):
            cost += np.linalg.norm(path[i+1] - path[i])
        success = True
        print(f"{name}: Success! Cost={cost:.2f}")
    else:
        print(f"{name}: Failed")


    num_ellipsoids = len(planner.ellipsoid_coords)
    if planner.ellipsoid_coords:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_title('Informed RRT* Ellipsoid Shrinkage (Evolution of Search Space)')
        
        # Plot Start/Goal
        ax.scatter(start[0], start[1], start[2], c='g', marker='o', s=150, label='Start')
        ax.scatter(goal[0], goal[1], goal[2], c='r', marker='x', s=150, label='Goal')
        ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], 'k:', alpha=0.5)

        # Plot the final path (if found)
        if path is not None:
             ax.plot(path[:, 0], path[:, 1], path[:, 2], 'g-', linewidth=3, label='Found Path')
        
        # Plot a subset of the recorded ellipsoids (e.g., first 5, middle 5, last 5)
        indices_to_plot = sorted(list(set([0, num_ellipsoids-1] + [int(i * (num_ellipsoids-1)/5) for i in range(1, 5)])))
        cmap = plt.cm.viridis
        
        for i, idx in enumerate(indices_to_plot):
            if idx < 0 or idx >= num_ellipsoids: continue
            params = planner.ellipsoid_coords[idx]
            
            plot_ellipsoid_3d(
                params, 
                ax=ax, 
                color=cmap(i / len(indices_to_plot)), 
                alpha=0.2 + 0.5 * (i / len(indices_to_plot))
            )

        # Plot the sampled nodes (optional)
        node_states = np.array([n.state for n in planner.node_list])
        ax.scatter(node_states[:, 0], node_states[:, 1], node_states[:, 2], c='gray', s=5, alpha=0.3, label='RRT* Nodes')

        plt.legend()
        plt.show()
    elif path is not None:
        print("Path found, but c_best was never improved from initial state.")
    else:
        print("No path found and no ellipsoid updates recorded.")
    return path, success, cost

def create_view(ax, bounds, start, goal, obstacles, results, colors, elev, azim, title):
    ax.clear()
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.view_init(elev=elev, azim=azim)
    
    # Draw Obstacles
    for obs in obstacles:
        if isinstance(obs, Sphere):
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
            x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
            y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
            z = obs.center[2] + obs.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # Draw Start and Goal
    ax.scatter(start[0], start[1], start[2], c='k', marker='o', s=100, label='Start', zorder=10)
    ax.scatter(goal[0], goal[1], goal[2], c='k', marker='*', s=100, label='Goal', zorder=10)
    
    # Draw Paths
    for name, res in results.items():
        path, success, cost = res
        if success and path is not None:
            ax.plot(path[:,0], path[:,1], path[:,2], color=colors[name], linewidth=2.5, 
                   label=f"{name} ({cost:.1f})", alpha=0.8)
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)

def main():
    # Environment Setup
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    results = {}
    
    # Run all planners
    path, success, cost = run_planner(BITStar, "BIT*", start, goal, obstacles, bounds, eta=2.5, batch_size=400)
    results["BIT*"] = (path, success, cost)
    
    path, success, cost = run_planner(RRTStar, "RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0)
    results["RRT*"] = (path, success, cost)

    path, success, cost = run_planner(InformedRRTStar, "Informed RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0)
    results["Informed RRT*"] = (path, success, cost)

    path, success, cost = run_planner(CustomRRTStar, "Custom RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0, clearance_weight=1.0)
    results["Custom RRT*"] = (path, success, cost)

    colors = {"BIT*": "blue", "RRT*": "red", "Informed RRT*": "green", "Custom RRT*": "orange"}
    
    # Create figure with multiple subplots for different views
    fig = plt.figure(figsize=(16, 12))
    
    # View 1: Default view (elevation=30, azimuth=45)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    create_view(ax1, bounds, start, goal, obstacles, results, colors, elev=30, azim=45, 
                title="View 1: Default (elev=30°, azim=45°)")
    
    # View 2: Top view
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    create_view(ax2, bounds, start, goal, obstacles, results, colors, elev=90, azim=0, 
                title="View 2: Top View (elev=90°)")
    
    # View 3: Side view
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    create_view(ax3, bounds, start, goal, obstacles, results, colors, elev=0, azim=0, 
                title="View 3: Side View (elev=0°, azim=0°)")
    
    # View 4: Front view
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    create_view(ax4, bounds, start, goal, obstacles, results, colors, elev=0, azim=90, 
                title="View 4: Front View (elev=0°, azim=90°)")
    
    # View 5: Isometric view
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    create_view(ax5, bounds, start, goal, obstacles, results, colors, elev=35, azim=135, 
                title="View 5: Isometric (elev=35°, azim=135°)")
    
    # View 6: Bottom-up view
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    create_view(ax6, bounds, start, goal, obstacles, results, colors, elev=-30, azim=225, 
                title="View 6: Bottom-Up (elev=-30°, azim=225°)")
    
    plt.tight_layout()
    plt.savefig('planner_comparison_multiview.png', dpi=150, bbox_inches='tight')
    print("\nMulti-view comparison saved to planner_comparison_multiview.png")

if __name__ == "__main__":
    main()
