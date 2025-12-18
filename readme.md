# Path Planning Project

<img src="media/gif/planner_trees_growth.gif" alt="Planner trees growth" width="640"/>

A  project implementing and comparing sampling-based motion planners in 3D: RRT, RRT*, Informed RRT*, Custom RRT*, and BIT*. The repository provides a unified runner (`main.py`) that executes planners, records convergence and metric data, and produces visualizations (convergence plots, metrics comparison, and tree+path overlays). The code is designed for reproducible experiments (seeded RNG) and flexible stopping modes (fixed iterations, time limit, or convergence).

**Quick Start**
- **Run:** `python3 main.py`

**Visualizations**
- **Animated growth:** `media/gif/planner_trees_growth.gif`
- **Tree + paths:** `media/img/time_or_iters_planner_trees_and_paths.png`
- **Metrics comparison:** `media/img/time_or_iters_metrics_comparison.png`
- **Convergence (best cost):** `media/img/time_or_iters_convergence_best_cost.png`

<!-- Embed examples (Markdown):
- Animated GIF: `![Planner trees growth](media/gif/planner_trees_growth.gif)`
- Static image: `![Trees and paths](media/img/time_or_iters_planner_trees_and_paths.png)` -->

<!-- Also show a static preview image right below the GIF -->
<img src="media/img/time_or_iters_planner_trees_and_paths.png" alt="Trees and paths" width="640"/>

**Repository Layout**
- **`main.py`**: experiment runner, visualization, and saving of metrics/images.
- **`src/common/`**: plotting and utility helpers (`extra_gif.py`, `extra_plot.py`, `utils.py`).
- **`src/planner/`**: planner implementations (`rrt.py`, `rrt_star.py`, `informed_rrt_star.py`, `custom_rrt_star.py`, `bit_star.py`).
- **`media/`**: saved metrics JSON, images, and gifs used for visualization.
- **`test/`**: scripts for generating animations and running comparisons.

**Notes & Tips**
- Toggle GIF generation by setting `plot_gif = True` in `main.py` (the script will attempt to generate and save GIFs under `media/gif/`).
- The runner supports three stop modes:
  - `iters` — stop after a fixed number of iterations,
  - `time_or_iters` — stop after a time limit or iterations,
  - `converged` — stop when best cost stops improving.
