import numpy as np

class Obstacle:
    def contains(self, point):
        raise NotImplementedError

class Sphere(Obstacle):
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def contains(self, point):
        return np.linalg.norm(point - self.center) <= self.radius

class Box(Obstacle):
    def __init__(self, center, extents):
        self.center = np.array(center)
        self.extents = np.array(extents) # half-extents

    def contains(self, point):
        p = np.array(point)
        return np.all(np.abs(p - self.center) <= self.extents)

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def sample_uniform(bounds, rng=None):
    """
    Uniform sample in bounds.
    If rng is provided, it must be a np.random.Generator.
    """
    if rng is None:
        return np.random.uniform(bounds[:, 0], bounds[:, 1])
    return rng.uniform(bounds[:, 0], bounds[:, 1])

def is_collision_free(p1, p2, obstacles, step_size=0.5):
    """
    Checks if line segment p1-p2 is collision-free.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = np.linalg.norm(p2 - p1)

    if dist < 1e-6:  # Points are essentially the same
        return True

    steps = int(np.ceil(dist / step_size))
    if steps == 0:
        steps = 1

    for i in range(steps + 1):
        t = i / steps
        p = p1 + t * (p2 - p1)
        for obs in obstacles:
            if obs.contains(p):
                return False
    return True
