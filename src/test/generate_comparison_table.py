import numpy as np
import time

from common.utils import Sphere, Box
from planners.bit_star import BITStar
from planners.rrt_star import RRTStar
from planners.informed_rrt_star import InformedRRTStar
from planners.custom_rrt_star import CustomRRTStar


def format_table(data, headers):
    """
    Format data as a simple ASCII table.
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Create separator
    separator = '+' + '+'.join(['-' * (w + 2) for w in col_widths]) + '+'
    
    # Format header
    header_row = '|' + '|'.join([f' {h:<{col_widths[i]}} ' for i, h in enumerate(headers)]) + '|'
    
    # Format data rows
    table_rows = []
    for row in data:
        table_row = '|' + '|'.join([f' {str(cell):<{col_widths[i]}} ' for i, cell in enumerate(row)]) + '|'
        table_rows.append(table_row)
    
    # Combine all parts
    table = [separator, header_row, separator]
    table.extend(table_rows)
    table.append(separator)
    
    return '\n'.join(table)


def run_planner_with_metrics(planner_class, name, start, goal, obstacles, bounds, **kwargs):
    """
    Run a planner and collect comprehensive metrics.
    
    Returns:
        dict: Dictionary containing all metrics for the planner
    """
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")
    
    planner = planner_class(start, goal, obstacles, bounds, **kwargs)
    
    start_time = time.time()
    path = planner.plan(max_time=15.0)  # 15 seconds budget per planner
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Initialize metrics
    metrics = {
        'Algorithm': name,
        'Success': False,
        'Path Length': float('inf'),
        'Completion Time (s)': duration,
        'Iterations/Trees': 0,
        'Number of Nodes': 0
    }
    
    # Get iteration count based on planner type
    if hasattr(planner, 'node_list'):  # RRT-based planners
        metrics['Iterations/Trees'] = len(planner.node_list)
        metrics['Number of Nodes'] = len(planner.node_list)
    elif hasattr(planner, 'V'):  # BIT* planner
        # BIT* uses batches, count total vertices explored
        metrics['Iterations/Trees'] = len(planner.V)
        metrics['Number of Nodes'] = len(planner.V)
    
    if path is not None:
        path = np.array(path)
        # Calculate path length cost
        path_length = 0
        for i in range(len(path)-1):
            path_length += np.linalg.norm(path[i+1] - path[i])
        
        metrics['Success'] = True
        metrics['Path Length'] = path_length
        
        print(f"✓ SUCCESS")
        print(f"  Path Length: {path_length:.4f}")
        print(f"  Time: {duration:.4f}s")
        print(f"  Nodes/Iterations: {metrics['Iterations/Trees']}")
    else:
        print(f"✗ FAILED - No path found within time limit")
        print(f"  Time: {duration:.4f}s")
        print(f"  Nodes explored: {metrics['Iterations/Trees']}")
    
    return metrics, path


def main():
    """
    Run all planners and generate a comprehensive comparison table.
    """
    print("\n" + "="*60)
    print("MOTION PLANNING ALGORITHMS COMPARISON")
    print("="*60)
    
    # Environment setup
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    
    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]
    
    print(f"\nEnvironment Configuration:")
    print(f"  Bounds: {bounds.tolist()}")
    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    print(f"  Obstacles: {len(obstacles)} spheres")
    
    # Store all metrics
    all_metrics = []
    all_paths = {}
    
    # Run BIT*
    metrics, path = run_planner_with_metrics(
        BITStar, "BIT*", start, goal, obstacles, bounds,
        eta=2.5, batch_size=400
    )
    all_metrics.append(metrics)
    all_paths["BIT*"] = path
    
    # Run RRT*
    metrics, path = run_planner_with_metrics(
        RRTStar, "RRT*", start, goal, obstacles, bounds,
        step_size=0.8, max_iter=3000, search_radius=3.0
    )
    all_metrics.append(metrics)
    all_paths["RRT*"] = path
    
    # Run Informed RRT*
    metrics, path = run_planner_with_metrics(
        InformedRRTStar, "Informed RRT*", start, goal, obstacles, bounds,
        step_size=0.8, max_iter=3000, search_radius=3.0
    )
    all_metrics.append(metrics)
    all_paths["Informed RRT*"] = path
    
    # Run Custom RRT*
    metrics, path = run_planner_with_metrics(
        CustomRRTStar, "Custom RRT*", start, goal, obstacles, bounds,
        step_size=0.8, max_iter=3000, search_radius=3.0, clearance_weight=1.0
    )
    all_metrics.append(metrics)
    all_paths["Custom RRT*"] = path
    
    # Generate comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60 + "\n")
    
    # Prepare table data
    table_data = []
    for m in all_metrics:
        row = [
            m['Algorithm'],
            '✓' if m['Success'] else '✗',
            f"{m['Path Length']:.4f}" if m['Success'] else "N/A",
            f"{m['Completion Time (s)']:.4f}",
            m['Iterations/Trees'],
            m['Number of Nodes']
        ]
        table_data.append(row)
    
    headers = [
        'Algorithm',
        'Success',
        'Path Length',
        'Time (s)',
        'Iterations',
        'Total Nodes'
    ]
    
    # Print table with custom formatter
    table = format_table(table_data, headers)
    print(table)
    
    # Save to file
    with open('comparison_table.txt', 'w') as f:
        f.write("MOTION PLANNING ALGORITHMS COMPARISON\n")
        f.write("="*60 + "\n\n")
        f.write(f"Environment Configuration:\n")
        f.write(f"  Bounds: {bounds.tolist()}\n")
        f.write(f"  Start: {start}\n")
        f.write(f"  Goal: {goal}\n")
        f.write(f"  Obstacles: {len(obstacles)} spheres\n\n")
        f.write(table)
        f.write("\n\n")
        
        # Additional statistics
        f.write("DETAILED METRICS:\n")
        f.write("="*60 + "\n\n")
        for m in all_metrics:
            f.write(f"{m['Algorithm']}:\n")
            for key, value in m.items():
                if key != 'Algorithm':
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    print(f"\n✓ Comparison table saved to 'comparison_table.txt'")
    
    # Print summary statistics
    successful_planners = [m for m in all_metrics if m['Success']]
    if successful_planners:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        best_path = min(successful_planners, key=lambda x: x['Path Length'])
        fastest = min(successful_planners, key=lambda x: x['Completion Time (s)'])
        most_efficient = min(successful_planners, key=lambda x: x['Iterations/Trees'])
        
        print(f"\n  Best Path Length: {best_path['Algorithm']} ({best_path['Path Length']:.4f})")
        print(f"  Fastest: {fastest['Algorithm']} ({fastest['Completion Time (s)']:.4f}s)")
        print(f"  Most Efficient: {most_efficient['Algorithm']} ({most_efficient['Iterations/Trees']} nodes)")
        print(f"\n  Success Rate: {len(successful_planners)}/{len(all_metrics)} algorithms")
    
    return all_metrics


if __name__ == "__main__":
    main()
