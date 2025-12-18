import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
N_SAMPLES = 5000       # Number of points to sample
FPS = 30               # Frames per second
DURATION = 10          # Total duration in seconds
RES = 150              # DPI for high resolution

# Define Problem Context (Start, Goal, and Costs)
x_start = np.array([-2.0, -2.0, -1.0])
x_goal  = np.array([2.0, 2.0, 3.0])

c_min = np.linalg.norm(x_goal - x_start)
c_best = c_min * 1.5  

# --- 1. Helper Functions ---

def sample_unit_ball(n):
    """
    Sample n points uniformly from a 3D unit ball.
    """
    points = []
    while len(points) < n:
        pt = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(pt) <= 1.0:
            points.append(pt)
    return np.array(points).T 

def get_rotation_matrix(start, goal):
    """
    Computes the rotation matrix C that aligns the x-axis [1,0,0]
    with the vector (goal - start).
    """
    fwd = goal - start
    fwd = fwd / np.linalg.norm(fwd)
    
    e1 = np.array([1.0, 0.0, 0.0])
    
    M = np.outer(fwd, e1)
    
    U, S, Vh = np.linalg.svd(M)
    
    d = np.linalg.det(np.dot(U, Vh))
    mid = np.eye(3)
    mid[2, 2] = d
    
    C = np.dot(np.dot(U, mid), Vh)
    return C

# --- 2. Precompute Geometry ---

# Step 1: Base samples (Unit Ball)
X_ball = sample_unit_ball(N_SAMPLES)

# --- NEW STEP 1.5: Generate Colors based on initial position ---
# We map the X-coordinate (which ranges from -1 to 1) to a color map (0 to 1).
# This creates a gradient along the axis that will be stretched and rotated.
# 'jet' or 'brg' are good colormaps because they have distinct ends.
colors = plt.cm.jet((X_ball[0, :] + 1) / 2) 

# Step 2: Calculate Scaling Matrix (L)
r1 = c_best / 2.0
r2 = np.sqrt(c_best**2 - c_min**2) / 2.0
r3 = r2
L = np.diag([r1, r2, r3])

# Step 3: Calculate Rotation Matrix (C)
C = get_rotation_matrix(x_start, x_goal)

# Step 4: Calculate Translation (Center)
x_center = (x_start + x_goal) / 2.0

# --- 3. Animation Setup ---

fig = plt.figure(figsize=(10, 8), dpi=RES)
ax = fig.add_subplot(111, projection='3d')

# Styling
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot Start and Goal for reference
ax.scatter([x_start[0]], [x_start[1]], [x_start[2]], color='black', s=100, marker='*', label='Start')
ax.scatter([x_goal[0]], [x_goal[1]], [x_goal[2]], color='black', s=100, marker='*', label='Goal')
ax.plot([x_start[0], x_goal[0]], [x_start[1], x_goal[1]], [x_start[2], x_goal[2]], 'k--', alpha=0.5)

# Initialize scatter with our pre-calculated colors
# Note: We do NOT change 'c' in the update loop, so the colors stick to the points
scat = ax.scatter(X_ball[0, :], X_ball[1, :], X_ball[2, :], c=colors, alpha=0.4, s=2)
txt = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

# --- 4. Animation Logic ---

total_frames = FPS * DURATION

def update(frame):
    progress = frame / total_frames
    
    stage = ""
    
    if progress < 0.15:
        # Phase 0: Unit Ball
        stage = "Sample Unit Ball"
        current_data = X_ball
        
    elif progress < 0.40:
        # Phase 1: Scale (Identity -> L)
        p = (progress - 0.15) / (0.40 - 0.15)
        scale_diag = (1 - p) * np.ones(3) + p * np.diag(L)
        current_L = np.diag(scale_diag)
        
        stage = "Scale"
        current_data = np.dot(current_L, X_ball)
        
    elif progress < 0.70:
        # Phase 2: Rotate
        p = (progress - 0.40) / (0.70 - 0.40)
        
        # Target points
        points_scaled = np.dot(L, X_ball)
        points_rotated = np.dot(C, points_scaled)
        
        # Linear interpolation of positions
        current_data = (1 - p) * points_scaled + p * points_rotated
        stage = "Rotate"
        
    else:
        # Phase 3: Translate
        p = (progress - 0.70) / (1.0 - 0.70)
        p = min(p, 1.0)
        
        points_oriented = np.dot(C, np.dot(L, X_ball))
        current_T = p * x_center
        current_data = points_oriented + current_T[:, np.newaxis]
        stage = "Move to Center"

    # Update scatter positions
    scat._offsets3d = (current_data[0, :], current_data[1, :], current_data[2, :])
    
    txt.set_text(stage)
    
    # Rotate camera
    ax.view_init(elev=20, azim=30 + frame * 0.2)
    
    return scat, txt

# --- 5. Save Video ---
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS, blit=False)

try:
    print("Generating animation...")
    writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("informed_rrt_colored.mp4", writer=writer)
    print("Saved as 'informed_rrt_colored.mp4'")
except Exception as e:
    print(f"Could not save video. Displaying instead. Error: {e}")
    plt.show()