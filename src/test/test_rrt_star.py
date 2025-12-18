import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.utils import Sphere, Box
from planners.rrt_star import RRTStar

def main():
    # Simpler Environment
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    # Fewer obstacles
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    # Planner
    print("Initializing RRT*...")
    planner = RRTStar(start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=2.0)
    
    print("Planning...")
    path = planner.plan(max_time=10.0)
    
    if path is not None:
        print("Path found!")
        path = np.array(path)
        cost = 0
        for i in range(len(path)-1):
            cost += np.linalg.norm(path[i+1] - path[i])
        print(f"Path cost: {cost:.2f}")
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
    
    # Draw Start and Goal
    ax.scatter(start[0], start[1], start[2], c='g', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='b', marker='*', s=100, label='Goal')
    
    # Draw Path
    if path is not None:
        ax.plot(path[:,0], path[:,1], path[:,2], 'g-', linewidth=3, label='Path')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    
    plt.savefig('rrt_star_test.png')
    print("Result saved to rrt_star_test.png")

if __name__ == "__main__":
    main()
