import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common.utils import Sphere, Box
from planners.bit_star import BITStar

def main():
    # Environment Setup
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.5),
        Box(center=[2, -2, 0], extents=[0.5, 0.5, 2]),
        Box(center=[-2, 2, 0], extents=[0.5, 0.5, 2]),
        Sphere(center=[-2, -2, 2], radius=0.8),
        Sphere(center=[2, 2, -2], radius=0.8)
    ]
    
    # Planner
    print("Initializing BIT*...")
    planner = BITStar(start, goal, obstacles, bounds, eta=1.5, batch_size=200)
    
    print("Planning...")
    path = planner.plan(max_time=10.0)
    
    if path is not None:
        print("Path found!")
        path = np.array(path)
    else:
        print("No path found.")

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    # Draw Obstacles
    for obs in obstacles:
        if isinstance(obs, Sphere):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
            y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
            z = obs.center[2] + obs.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="r", alpha=0.5)
        elif isinstance(obs, Box):
            # Simple wireframe for box (just corners/edges would be cleaner, but scatter is easy)
            # Let's draw center for now or basic wireframe
            c = obs.center
            e = obs.extents
            # Draw a point for box center
            ax.scatter(c[0], c[1], c[2], color="r", marker='s', s=100)
            # Can add full cube drawing if needed, but simple marker is okay for now
    
    # Draw Start and Goal
    ax.scatter(start[0], start[1], start[2], c='g', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='b', marker='*', s=100, label='Goal')
    
    # Draw Tree (Optional, busy plot)
    # for node in planner.V:
    #     if node.parent:
    #         p1 = node.state
    #         p2 = node.parent.state
    #         ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', alpha=0.1)

    # Draw Path
    if path is not None:
        ax.plot(path[:,0], path[:,1], path[:,2], 'g-', linewidth=3, label='Path')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    
    plt.savefig('bit_star_result.png')
    print("Result saved to bit_star_result.png")

if __name__ == "__main__":
    main()
