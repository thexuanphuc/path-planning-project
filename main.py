import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from common.utils import Sphere, Box
from planner.bit_star import BITStar
from planner.rrt_star import RRTStar
from planner.informed_rrt_star import InformedRRTStar
from planner.custom_rrt_star import CustomRRTStar
from planner.rrt import RRT



def _path_length(path):
    if path is None:
        return 0.0
    path = np.asarray(path)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))


def run_planner(PlannerClassOrFactory, planner_name, start, goal, obstacles, bounds, **planner_kwargs):
    """
    Unified experiment runner.

    Required stop_mode ∈ {"iters", "time_or_iters", "converged"}.

    Returns a dict with required metrics:
      path, success, path_length, total_cost, nodes_in_tree,
      planning_time, iterations, time_to_first_solution,
      best_cost_history
    """
    stop_mode = planner_kwargs.pop("stop_mode")
    seed = int(planner_kwargs.pop("seed", 0))

    max_iters = planner_kwargs.pop("max_iters", None)
    time_limit = planner_kwargs.pop("time_limit", None)

    eps_abs = float(planner_kwargs.pop("eps_abs", 1e-3))
    eps_rel = float(planner_kwargs.pop("eps_rel", 1e-3))
    patience = int(planner_kwargs.pop("patience", 10))
    check_every = int(planner_kwargs.pop("check_every", 25))

    # --- deterministic seeding across numpy + python random ---
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # (optional safety) if any legacy code still uses global np.random
    np.random.seed(seed)

    # Build planner instance
    if callable(PlannerClassOrFactory):
        planner = PlannerClassOrFactory(
            start, goal, obstacles, bounds,
            rng=rng,
            **planner_kwargs
        )
    else:
        raise ValueError("PlannerClassOrFactory must be a class or factory callable.")

    t0 = time.perf_counter()
    iterations = 0

    best_cost = float("inf")
    best_cost_history = []
    time_to_first_solution = None
    success = False
    best_path = None
    total_cost = float("inf")

    no_improve_checks = 0

    def should_stop():
        nonlocal no_improve_checks, best_cost
        if stop_mode == "iters":
            return (max_iters is not None) and (iterations >= max_iters)

        if stop_mode == "time_or_iters":
            if (time_limit is not None) and ((time.perf_counter() - t0) >= time_limit):
                return True
            if (max_iters is not None) and (iterations >= max_iters):
                return True
            return False

        if stop_mode == "converged":
            # stop when no improvement for 'patience' checks
            return no_improve_checks >= patience

        raise ValueError(f"Unknown stop_mode: {stop_mode}")

    # Main controlled loop
    while True:
        if should_stop():
            break

        # For mode "converged", we ignore iter/time caps as per spec:
        # wrapper still needs to keep looping until convergence triggers.
        if stop_mode == "converged":
            pass
        else:
            # for other modes, enforce max_iters if provided
            if (max_iters is not None) and (iterations >= max_iters):
                break
            if (stop_mode == "time_or_iters") and (time_limit is not None) and ((time.perf_counter() - t0) >= time_limit):
                break

        # one incremental step
        planner.step()
        iterations += 1

        # update current best solution
        path, ok, cost = planner.get_best_solution()
        now = time.perf_counter()
        elapsed = now - t0

        if ok and path is not None:
            if not success:
                success = True
                time_to_first_solution = elapsed
            best_path = np.asarray(path)
            total_cost = float(cost)

        # track best-cost for convergence + plots
        current_best = float(cost) if ok else float("inf")
        # best_cost is "c_best"
        if current_best < best_cost:
            best_cost = current_best

        best_cost_history.append((elapsed, iterations, best_cost))

        # convergence checks only every K iterations
        if stop_mode == "converged" and (iterations % check_every == 0):
            # recompute current best (already in best_cost)
            # improvement test: improved by eps_abs OR eps_rel
            prev = best_cost_history[-check_every][2] if len(best_cost_history) > check_every else float("inf")
            new = best_cost

            improved = False
            if prev == float("inf") and new < float("inf"):
                improved = True
            elif prev < float("inf") and new < float("inf"):
                if (prev - new) >= eps_abs:
                    improved = True
                elif (prev - new) >= eps_rel * abs(prev):
                    improved = True

            if improved:
                no_improve_checks = 0
            else:
                no_improve_checks += 1

    planning_time = time.perf_counter() - t0

    nodes_in_tree = None
    if hasattr(planner, "node_list"):
        nodes_in_tree = len(planner.node_list)
    elif hasattr(planner, "nodes_in_tree"):
        nodes_in_tree = int(planner.nodes_in_tree)

    result = {
        "planner": planner_name,
        "stop_mode": stop_mode,
        "seed": seed,

        "path": None if best_path is None else best_path.tolist(),
        "success": bool(success),

        "path_length": float(_path_length(best_path)) if best_path is not None else float("inf" if not success else 0.0),
        "total_cost": float(total_cost if success else float("inf")),

        "nodes_in_tree": int(nodes_in_tree) if nodes_in_tree is not None else None,
        "planning_time": float(planning_time),
        "iterations": int(iterations),
        "time_to_first_solution": None if time_to_first_solution is None else float(time_to_first_solution),

        "best_cost_history": [(float(t), int(it), float(c)) for (t, it, c) in best_cost_history],
    }
    return result


def main():
    # Environment (kept intact)
    bounds = np.array([[-5, 5], [-5, 5], [-5, 5]])
    start = [-4, -4, -4]
    goal = [4, 4, 4]

    obstacles = [
        Sphere(center=[0, 0, 0], radius=1.2),
        Sphere(center=[-2, -2, 2], radius=0.6),
        Sphere(center=[2, 2, -2], radius=0.6)
    ]

    planners = [
        (RRT, "RRT", dict(step_size=0.8, max_iter=3000, goal_sample_rate=0.2, goal_tolerance=2.0)),
        (BITStar, "BIT*", dict(eta=2.5, batch_size=400)),
        (RRTStar, "RRT*", dict(step_size=0.8, max_iter=3000, search_radius=3.0)),
        (InformedRRTStar, "Informed RRT*", dict(step_size=0.8, max_iter=3000, search_radius=3.0)),
        (CustomRRTStar, "Custom RRT*", dict(step_size=0.8, max_iter=3000, search_radius=3.0, clearance_weight=1.0)),
    ]

    stop_modes = [
        # # Mode A — Fixed iterations (∞ time)
        # dict(stop_mode="iters", max_iters=3000),

        # Mode B — Time limit + iteration cap
        dict(stop_mode="time_or_iters", max_iters=3000, time_limit=15.0),

        # # Mode C — Cost convergence only
        # dict(stop_mode="converged", eps_abs=1e-3, eps_rel=1e-3, patience=10, check_every=25),
    ]

    seed = 0

    all_results = []
    for Planner, name, base_kwargs in planners:
        for sm in stop_modes:
            kwargs = dict(base_kwargs)
            kwargs.update(sm)

            # normalize naming: your planners currently use max_iter internally;
            # wrapper uses max_iters for stopping. planner constructor still takes max_iter.
            # so we keep constructor kwargs as-is, and wrapper stopping is controlled by max_iters/time_limit.
            kwargs["seed"] = seed

            print(f"\n--- {name} | stop_mode={sm['stop_mode']} | seed={seed} ---")
            res = run_planner(Planner, name, start, goal, obstacles, bounds, **kwargs)
            print(f"  success={res['success']}, total_cost={res['total_cost']:.4f}, "
                  f"len={res['path_length']:.4f}, iters={res['iterations']}, "
                  f"time={res['planning_time']:.3f}s, t_first={res['time_to_first_solution']}")
            all_results.append(res)

    # Save metrics
    with open("planner_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved metrics to planner_metrics.json")

    # Optional: convergence plots (best_cost_history)
    plt.figure()
    for res in all_results:
        hist = res["best_cost_history"]
        if not hist:
            continue
        t = [h[0] for h in hist]
        c = [h[2] for h in hist]
        label = f"{res['planner']}|{res['stop_mode']}"
        plt.plot(t, c, label=label)
    plt.xlabel("time (s)")
    plt.ylabel("best cost")
    plt.title("Convergence (best cost vs time)")
    plt.legend()
    plt.savefig("convergence_best_cost.png")
    print("Saved convergence plot to convergence_best_cost.png")

    # Visualization (kept very close to your original)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])

    for obs in obstacles:
        if isinstance(obs, Sphere):
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
            x = obs.center[0] + obs.radius * np.cos(u) * np.sin(v)
            y = obs.center[1] + obs.radius * np.sin(u) * np.sin(v)
            z = obs.center[2] + obs.radius * np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)
        elif isinstance(obs, Box):
            ax.scatter(obs.center[0], obs.center[1], obs.center[2], color="gray", marker='s', s=50)

    ax.scatter(start[0], start[1], start[2], c='k', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], goal[2], c='k', marker='*', s=100, label='Goal')

    # draw only the "best" run per planner (prefer converged, else time_or_iters, else iters)
    pref = {"converged": 0, "time_or_iters": 1, "iters": 2}
    best_per_planner = {}
    for res in all_results:
        key = res["planner"]
        if (key not in best_per_planner) or (pref[res["stop_mode"]] < pref[best_per_planner[key]["stop_mode"]]):
            best_per_planner[key] = res

    colors = {"BIT*": "blue", "RRT*": "red", "Informed RRT*": "green", "Custom RRT*": "orange"}
    for name, res in best_per_planner.items():
        if res["success"] and res["path"] is not None:
            path = np.array(res["path"])
            ax.plot(path[:, 0], path[:, 1], path[:, 2],
                    color=colors.get(name, "black"),
                    linewidth=2,
                    label=f"{name} ({res['stop_mode']}, C:{res['total_cost']:.1f}, T:{res['planning_time']:.1f}s)")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Comparison of Sampling-Based Planners (best run per planner)")
    plt.legend()
    plt.savefig("planner_comparison.png")
    print("Saved comparison plot to planner_comparison.png")


if __name__ == "__main__":
    main()
