import json
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import animation
from common.utils import Sphere, Box
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _load_results(path="media/planner_metrics.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def _prepare_planner_data(res):
    # nodes is list of [x,y,z]
    raw_nodes = res.get("nodes", []) or []
    raw_edges = res.get("edges", []) or []
    raw_path = res.get("path", []) or []

    # robustly parse nodes: accept only entries that can be converted to 3 floats
    parsed_nodes = []
    for item in raw_nodes:
        try:
            # allow sequences longer than 3 (take first 3) or exactly 3
            coords = list(item)
            if len(coords) >= 3:
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    parsed_nodes.append([x, y, z])
        except Exception:
            continue
    if parsed_nodes:
        nodes = np.asarray(parsed_nodes, dtype=float)
    else:
        nodes = np.zeros((0, 3), dtype=float)

    # map node coordinate tuple to index (first occurrence)
    coord_to_idx = {tuple(p): i for i, p in enumerate(nodes.tolist())}

    # parse edges: keep only edges whose endpoints exist in nodes
    edges_idx = []
    segments = []
    for e in raw_edges:
        try:
            p, q = e
            pp = (float(p[0]), float(p[1]), float(p[2]))
            qq = (float(q[0]), float(q[1]), float(q[2]))
        except Exception:
            continue
        if pp in coord_to_idx and qq in coord_to_idx:
            a = coord_to_idx[pp]
            b = coord_to_idx[qq]
            edges_idx.append((a, b))
            pa = nodes[a]
            pb = nodes[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                segments.append(np.vstack([pa, pb]))

    # parse path: accept sequence of 3-float points
    parsed_path = []
    for item in raw_path:
        try:
            coords = list(item)
            if len(coords) >= 3:
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    parsed_path.append([x, y, z])
        except Exception:
            continue
    if parsed_path:
        path = np.asarray(parsed_path, dtype=float)
    else:
        path = np.zeros((0, 3), dtype=float)
    return {
        "name": res.get("planner", "planner"),
        "nodes": nodes,
        "edges_idx": edges_idx,
        "segments": segments,
        "path": path,
        "stop_mode": res.get("stop_mode", ""),
    }


def make_growth_gif(results_path="media/planner_metrics.json", bounds=None, start=None, goal=None, obstacles=None,
                    out_file="media/gif/planner_trees_growth.gif", max_frames=200, fps=20):
    """Create a GIF showing node-addition growth for each planner on a 2x2 subplot grid.

    - results_path: path to JSON produced by `main.py`
    - bounds/start/goal/obstacles: optional environment info for plotting
    """
    print("start make_growth_gif()")
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

    if not prepared:
        print("No planner results found to animate (prepared is empty).")
        return

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
    # Prepare and cache artists for each subplot to avoid heavy create/remove
    for i, p in enumerate(prepared):
        ax = axes[i]
        # initial empty scatter
        pts = p["nodes"]
        if pts.size:
            sc = ax.scatter([], [], [], s=8, alpha=0.12, color='blue')
        else:
            sc = ax.scatter([], [], [], s=8, alpha=0.12, color='blue')

        # Line3DCollection for edges (start empty)
        segs = p.get("segments", [])
        if segs:
            coll = Line3DCollection([], colors='blue', linewidths=0.5, alpha=0.04)
            ax.add_collection3d(coll)
        else:
            coll = Line3DCollection([], colors='blue', linewidths=0.5, alpha=0.04)
            ax.add_collection3d(coll)

        # path line (hidden until final frame)
        path_line = None
        if p["path"].size:
            line, = ax.plot([], [], [], color='red', linewidth=2.0)
            path_line = line

        # small text label for stop_mode
        txt = ax.text2D(0.05, 0.92, f"stop_mode: {p.get('stop_mode','')}", transform=ax.transAxes, fontsize=8)

        # store artist handles
        p['artists'] = {
            'scatter': sc,
            'edges_coll': coll,
            'path_line': path_line,
            'text': txt,
        }

    # animation update using cached artists
    def update(frame_i):
        k = int(idxs[frame_i])
        artists = []
        for i, p in enumerate(prepared):
            ax = axes[i]
            pts = p['nodes']
            a = p['artists']

            # update scatter
            if pts.size and k > 0:
                upto = min(k, pts.shape[0])
                pts_shown = pts[:upto]
                try:
                    a['scatter']._offsets3d = (pts_shown[:, 0], pts_shown[:, 1], pts_shown[:, 2])
                except Exception:
                    # fallback: remove and recreate small scatter
                    try:
                        a['scatter'].remove()
                    except Exception:
                        pass
                    a['scatter'] = ax.scatter(pts_shown[:, 0], pts_shown[:, 1], pts_shown[:, 2], s=8, alpha=0.12, color='blue')
            else:
                # empty scatter
                try:
                    a['scatter']._offsets3d = ([], [], [])
                except Exception:
                    pass

            # update edges collection: include only edges where both endpoints < upto
            segs = p.get('segments', [])
            if segs and pts.size and k > 0:
                upto = min(k, pts.shape[0])
                # build segments_upto by filtering edges_idx
                edges_idx = p.get('edges_idx', [])
                segs_upto = [segs[j] for j, (ai, bi) in enumerate(edges_idx) if ai < upto and bi < upto]
                try:
                    a['edges_coll'].set_segments(segs_upto)
                except Exception:
                    # fallback: recreate collection
                    try:
                        a['edges_coll'].remove()
                    except Exception:
                        pass
                    a['edges_coll'] = Line3DCollection(segs_upto, colors='blue', linewidths=0.5, alpha=0.04)
                    ax.add_collection3d(a['edges_coll'])
            else:
                try:
                    a['edges_coll'].set_segments([])
                except Exception:
                    pass

            # overlay final path on last frame
            if frame_i == len(idxs) - 1 and p['path'].size:
                path = p['path']
                if a.get('path_line') is None:
                    # create if missing
                    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=2.0)
                    a['path_line'] = line
                else:
                    try:
                        a['path_line'].set_data_3d(path[:, 0], path[:, 1], path[:, 2])
                    except Exception:
                        try:
                            a['path_line'].remove()
                        except Exception:
                            pass
                        a['path_line'], = ax.plot(path[:, 0], path[:, 1], path[:, 2], color='red', linewidth=2.0)

            artists.extend([a['scatter'], a['edges_coll'], a.get('path_line'), a['text']])

        fig.suptitle('Planner tree growth (nodes added over time)')
        # return list of artists (None entries filtered)
        return [art for art in artists if art is not None]

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
        results = _load_results('media/planner_metrics.json')
    except Exception:
        print('media/planner_metrics.json not found in cwd; please run main.py first')
        results = []

    # reconstruct basic env used in the repo's main (fallback)
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]
    obstacles = [Sphere(center=[0, 0, 0], radius=1.2), Sphere(center=[-2, -2, 2], radius=0.6), Sphere(center=[2, 2, -2], radius=0.6)]

    # Do not run animation automatically here. Save directly into `media/gif/` by
    # calling `make_growth_gif(...)` from your driver (for example, `main.py`).
    print('extra_gif module loaded. To create GIFs call make_growth_gif(...) from your script.')
