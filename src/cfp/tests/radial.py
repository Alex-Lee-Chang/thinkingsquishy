import numpy as np
from matplotlib import pyplot as plt
import math

center = (0.5,0.4)
radius = 0.08 # Max radius of 0.17

with open("shape.txt", 'r') as f:
    lines = f.readlines()
    points_np = np.array([list(map(float, line.strip().split())) for line in lines])

# Generate filtered points
distances = np.linalg.norm(points_np - center, axis=1)
filtered = points_np[distances > radius]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

graph1 = axes[0]
graph1.plot(points_np[:, 0], points_np[:, 1], 'o', markersize=2)

graph2 = axes[1]
graph2.plot(filtered[:, 0], filtered[:, 1], 'o', markersize=2)

plt.show()