import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from planners.rrt_star import RRTStar

def main():
    # NO OBSTACLES - just test basic functionality
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = []  # EMPTY!
    
    # Planner
    print("Initializing RRT* (no obstacles)...")
    planner = RRTStar(start, goal, obstacles, bounds, step_size=1.0, max_iter=1000, search_radius=3.0)
    
    print("Planning...")
    path = planner.plan(max_time=5.0)
    
    if path is not None:
        print("Path found!")
        path = np.array(path)
        cost = 0
        for i in range(len(path)-1):
            cost += np.linalg.norm(path[i+1] - path[i])
        print(f"Path cost: {cost:.2f}, Path length: {len(path)}")
    else:
        print("No path found - THIS IS A BUG!")

    # Visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    # Draw Start and Goal
    ax.scatter(start[0], start[1], start[2], c='g', marker='o', s=200, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='b', marker='*', s=200, label='Goal')
    
    # Draw Path
    if path is not None:
        ax.plot(path[:,0], path[:,1], path[:,2], 'g-', linewidth=3, label='Path')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    
    plt.savefig('minimal_test.png')
    print("Result saved to minimal_test.png")

if __name__ == "__main__":
    main()
