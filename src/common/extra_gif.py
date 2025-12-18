import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation
from common.utils import Sphere, Box


def _load_results(path="planner_metrics.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _prepare_planner_data(res):
    # nodes is list of [x,y,z]
    nodes = np.asarray(res.get("nodes", []), dtype=float)
    edges = res.get("edges", [])
    # map node coordinate tuple to index (first occurrence)
    coord_to_idx = {}
    for i, p in enumerate(nodes.tolist()):
        coord_to_idx[tuple(map(float, p))] = i

    edges_idx = []
    for (p, q) in edges:
        tp = tuple(map(float, p))
        tq = tuple(map(float, q))
        if tp in coord_to_idx and tq in coord_to_idx:
            edges_idx.append((coord_to_idx[tp], coord_to_idx[tq]))

    path = np.asarray(res.get("path", []), dtype=float)
    return {
        "name": res.get("planner", "planner"),
        "nodes": nodes,
        "edges_idx": edges_idx,
        "path": path,
        "stop_mode": res.get("stop_mode", ""),
    }


def make_growth_gif(results_path="planner_metrics.json", bounds=None, start=None, goal=None, obstacles=None,
                    out_file="media/gif/planner_trees_growth.gif", max_frames=200, fps=12):
    """Create a GIF showing node-addition growth for each planner on a 2x2 subplot grid.

    - results_path: path to JSON produced by `main.py`
    - bounds/start/goal/obstacles: optional environment info for plotting
    """
    results = _load_results(results_path)

    # keep only the first 4 unique planners (in order encountered)
    unique = []
    for r in results:
        if r["planner"] not in unique:
            unique.append(r["planner"])
    # filter results to first occurrence per planner
    planners_res = []
    seen = set()
    for r in results:
        if r["planner"] in seen:
            continue
        seen.add(r["planner"])
        planners_res.append(r)
        if len(planners_res) >= 4:
            break

    prepared = [_prepare_planner_data(r) for r in planners_res]

    # determine frame count by max nodes across planners
    max_nodes = max((p["nodes"].shape[0] for p in prepared), default=1)
    if max_nodes <= 0:
        raise RuntimeError("No nodes found in planner results to animate.")

    # limit frames
    frames = min(max_frames, max_nodes)
    # sample indices that grow from 1..max_nodes
    idxs = np.unique(np.round(np.linspace(1, max_nodes, frames)).astype(int))

    # prepare figure
    n = len(prepared)
    cols = 2
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    axes = []
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        axes.append(ax)
        if bounds is not None:
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
            ax.set_zlim(bounds[2])
        ax.set_title(prepared[i]["name"])

    # static drawing of obstacles and start/goal
    def _draw_static(ax):
        if obstacles:
            for obs in obstacles:
                if isinstance(obs, Sphere):
                    u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:8j]
                    x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
                    y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
                    z = obs.center[2] + obs.radius * np.cos(v)
                    ax.plot_wireframe(x, y, z, color="gray", alpha=0.25)
                elif isinstance(obs, Box):
                    ax.scatter(obs.center[0], obs.center[1], obs.center[2], color="gray", marker='s', s=40)
        if start is not None:
            ax.scatter(start[0], start[1], start[2], c='k', marker='o', s=60)
        if goal is not None:
            ax.scatter(goal[0], goal[1], goal[2], c='k', marker='*', s=60)

    for ax in axes:
        _draw_static(ax)

    # animation update
    def update(frame_i):
        k = idxs[frame_i]
        for i, p in enumerate(prepared):
            ax = axes[i]
            # remove previous plotted artists (collections, lines, texts)
            # (assignment like `ax.lines = []` is not allowed on some Matplotlib versions)
            for coll in list(ax.collections):
                try:
                    coll.remove()
                except Exception:
                    pass
            for line in list(ax.lines):
                try:
                    line.remove()
                except Exception:
                    pass
            for txt in list(ax.texts):
                try:
                    txt.remove()
                except Exception:
                    pass

            # redraw static
            _draw_static(ax)

            nodes = p["nodes"]
            if nodes.size:
                upto = min(k, nodes.shape[0])
                pts = nodes[:upto]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, alpha=0.12, color='blue')

                # draw edges where both endpoints index < upto
                for (a_idx, b_idx) in p["edges_idx"]:
                    if a_idx < upto and b_idx < upto:
                        pa = nodes[a_idx]
                        pb = nodes[b_idx]
                        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color='blue', alpha=0.04, linewidth=0.5)

            # if last frame or all nodes shown, overlay final path
            if frame_i == len(idxs) - 1 and p["path"].size:
                path = p["path"]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=2.0)

            # annotate with stop_mode small label
            ax.text2D(0.05, 0.92, f"stop_mode: {p.get('stop_mode','')}", transform=ax.transAxes, fontsize=8)

        fig.suptitle('Planner tree growth (nodes added over time)')
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(idxs), interval=1000 // fps)

    # save
    out_file = out_file
    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(out_file, writer=writer)
        print(f"Saved GIF to {out_file}")
    except Exception as e:
        print("Failed to save GIF via PillowWriter:", e)
        # fallback: save as mp4 if ffmpeg available
        try:
            writer = animation.FFMpegWriter(fps=fps)
            mp4file = out_file.replace('.gif', '.mp4')
            ani.save(mp4file, writer=writer)
            print(f"Saved MP4 to {mp4file}")
        except Exception as e2:
            print("Failed to save animation:", e2)


# Note: per user preference, this module will not create directories or copy files.
# It will write the GIF only to the provided `out_file` path; ensure the target
# directory exists before calling.


if __name__ == '__main__':
    # Run from repo root with PYTHONPATH=src
    # Try to load env info from main.py run (best-effort)
    res = None
    try:
        results = _load_results('planner_metrics.json')
    except Exception:
        print('planner_metrics.json not found in cwd; please run main.py first')
        results = []

    # reconstruct basic env used in the repo's main (fallback)
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    obstacles = [Sphere(center=[0, 0, 0], radius=1.2), Sphere(center=[-2, -2, 2], radius=0.6), Sphere(center=[2, 2, -2], radius=0.6)]

    # Do not run animation automatically here. Save directly into `media/gif/` by
    # calling `make_growth_gif(...)` from your driver (for example, `main.py`).
    print('extra_gif module loaded. To create GIFs call make_growth_gif(...) from your script.')
