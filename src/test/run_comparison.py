import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from common.utils import Sphere, Box
from planners.bit_star import BITStar
from planners.rrt_star import RRTStar
from planners.informed_rrt_star import InformedRRTStar
from planners.custom_rrt_star import CustomRRTStar

def run_planner(planner_class, name, start, goal, obstacles, bounds, **kwargs):
    print(f"--- Running {name} ---")
    planner = planner_class(start, goal, obstacles, bounds, **kwargs)
    
    start_time = time.time()
    path = planner.plan(max_time=15.0) # 15 seconds budget per planner
    end_time = time.time()
    
    duration = end_time - start_time
    cost = float('inf')
    success = False
    
    if path is not None:
        path = np.array(path)
        # Calculate path length cost
        cost = 0
        for i in range(len(path)-1):
            cost += np.linalg.norm(path[i+1] - path[i])
        success = True
        print(f"{name}: Success! Cost={cost:.2f}, Time={duration:.2f}s")
    else:
        print(f"{name}: Failed within time limit.")
        
    return path, success, cost, duration

def main():
    # Simpler Environment for successful demonstration
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    results = {}
    
    # # Run BIT*
    # path, success, cost, duration = run_planner(BITStar, "BIT*", start, goal, obstacles, bounds, eta=2.5, batch_size=400)
    # results["BIT*"] = (path, success, cost, duration)
    
    # # Run RRT*
    # path, success, cost, duration = run_planner(RRTStar, "RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0)
    # results["RRT*"] = (path, success, cost, duration)

    # Run Informed RRT*
    path, success, cost, duration = run_planner(InformedRRTStar, "Informed RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0)
    results["Informed RRT*"] = (path, success, cost, duration)

    # # Run Custom RRT*
    # path, success, cost, duration = run_planner(CustomRRTStar, "Custom RRT*", start, goal, obstacles, bounds, step_size=0.8, max_iter=3000, search_radius=3.0, clearance_weight=1.0)
    # results["Custom RRT*"] = (path, success, cost, duration)

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    
    # Draw Obstacles (same drawing logic as before)
    for obs in obstacles:
        if isinstance(obs, Sphere):
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
            x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
            y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
            z = obs.center[2] + obs.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)
        elif isinstance(obs, Box):
            ax.scatter(obs.center[0], obs.center[1], obs.center[2], color="gray", marker='s', s=50)

    # Draw Start and Goal
    ax.scatter(start[0], start[1], start[2], c='k', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='k', marker='*', s=100, label='Goal')
    
    colors = {"BIT*": "blue", "RRT*": "red", "Informed RRT*": "green", "Custom RRT*": "orange"}
    
    for name, res in results.items():
        path, success, cost, duration = res
        if success and path is not None:
            ax.plot(path[:,0], path[:,1], path[:,2], color=colors[name], linewidth=2, label=f"{name} (C:{cost:.1f}, T:{duration:.1f}s)")
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Comparison of Sampling-Based Planners")
    plt.legend()
    
    plt.savefig('planner_comparison.png')
    print("Comparison saved to planner_comparison.png")

if __name__ == "__main__":
    main()
