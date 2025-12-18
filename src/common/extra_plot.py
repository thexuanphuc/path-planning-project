import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _extract_nodes_and_edges(planner):
    """Return (nodes: (N,3) array, edges: list of (p,q) pairs) for many planner types."""
    nodes = []
    edges = []

    # RRT / RRTStar / CustomRRTStar style
    if hasattr(planner, "node_list"):
        for n in planner.node_list:
            try:
                s = np.asarray(n.state)
            except Exception:
                continue
            nodes.append(s)
            if hasattr(n, "parent") and n.parent is not None:
                try:
                    edges.append((s, np.asarray(n.parent.state)))
                except Exception:
                    pass

    # BIT* style
    if hasattr(planner, "V") and planner.V is not None:
        for v in planner.V:
            try:
                s = np.asarray(v.state)
            except Exception:
                continue
            nodes.append(s)
            if hasattr(v, "parent") and v.parent is not None:
                try:
                    edges.append((s, np.asarray(v.parent.state)))
                except Exception:
                    pass

    if not nodes:
        return np.zeros((0, 3)), []

    arr = np.vstack(nodes)
    return arr, edges


def plot_tree_3d(ax, planner, color="gray", node_size=6, node_alpha=0.12, edge_alpha=0.05):
    """Plot planner nodes and tree edges onto provided 3D Axes.

    - `planner` is the planner instance (RRT, RRTStar, BITStar, ...)
    - returns number of nodes plotted
    """
    nodes, edges = _extract_nodes_and_edges(planner)
    if nodes.size == 0:
        return 0

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=color, s=node_size, alpha=node_alpha)

    # draw thin edges
    for p, q in edges:
        xs = [p[0], q[0]]
        ys = [p[1], q[1]]
        zs = [p[2], q[2]]
        ax.plot(xs, ys, zs, color=color, linewidth=0.5, alpha=edge_alpha)

    return nodes.shape[0]


def plot_path_3d(ax, path, color="red", linewidth=2.0, zorder=10):
    if path is None:
        return
    p = np.asarray(path)
    if p.size == 0:
        return
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=linewidth, zorder=zorder)


def plot_metrics_comparison(all_results, out_file="metrics_comparison.png"):
    """Create bar charts comparing core model metrics across planner.

    Metrics: path_length, planning_time, iterations, nodes_in_tree
    """
    import matplotlib.pyplot as plt

    labels = [f"{r['planner']} ({r['stop_mode']})" for r in all_results]
    path_len = [r.get('path_length', float('nan')) for r in all_results]
    time_req = [r.get('planning_time', float('nan')) for r in all_results]
    iters = [r.get('iterations', float('nan')) for r in all_results]
    nodes = [r.get('nodes_in_tree', float('nan')) for r in all_results]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].bar(labels, path_len)
    axs[0].set_title('Path length')
    axs[0].tick_params(axis='x', rotation=45)

    axs[1].bar(labels, time_req)
    axs[1].set_title('Planning time (s)')
    axs[1].tick_params(axis='x', rotation=45)

    axs[2].bar(labels, iters)
    axs[2].set_title('Iterations')
    axs[2].tick_params(axis='x', rotation=45)

    axs[3].bar(labels, nodes)
    axs[3].set_title('Nodes in tree')
    axs[3].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)
    return out_file
