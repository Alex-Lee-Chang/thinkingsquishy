import numpy as np
import math
from collections import defaultdict


with open("shape.txt", 'r') as f:
        lines = f.readlines()
        points_np = np.array([list(map(float, line.strip().split())) for line in lines])

grid_size = .005

xmin, ymin = points_np.min(axis=0)
xmax, ymax = points_np.max(axis=0)

grid_x_size = math.ceil((xmax - xmin) / grid_size) + 2
grid_y_size = math.ceil((ymax - ymin) / grid_size) + 2
grid = np.empty((grid_x_size, grid_y_size), dtype=object)

grid_indices = (((points_np - [xmin, ymin]) // grid_size) + 1).astype(int)

grid = defaultdict(list)

for i, gridcoords in enumerate(grid_indices):

    grid_indices_x = gridcoords[0]
    grid_indices_y = gridcoords[1]

    key = grid_indices_x + grid_indices_y * grid_x_size 
    grid[key].append(i)

boundary_cells = []

for cell in grid:
    if (cell-1 not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell+1)
    elif (cell+1 not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell-1)
    elif (cell-grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell+grid_x_size)
    elif (cell+grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell-grid_x_size)
    elif (cell-1-grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell+grid_x_size+1)
    elif (cell+1-grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell+grid_x_size-1)
    elif (cell-1+grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell-grid_x_size-1)
    elif (cell+1+grid_x_size not in grid):
        boundary_cells.append(cell)
        boundary_cells.append(cell-grid_x_size+1)


boundary_points = np.concatenate([points_np[grid[cell]] for cell in boundary_cells], axis=0)

from matplotlib import pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

graph1 = axes[0]
graph1.plot(points_np[:, 0], points_np[:, 1], 'o', markersize=2)

graph2 = axes[1]
graph2.plot(boundary_points[:, 0], boundary_points[:, 1], 'o', markersize=2)


# Grid visualization

x_lines = np.arange(xmin-2*grid_size, xmax + 2*grid_size, grid_size)
y_lines = np.arange(ymin-2*grid_size, ymax + 2*grid_size, grid_size)

for x in x_lines:
    plt.plot([x, x], [ymin-2*grid_size, ymax+2*grid_size], color='gray', linewidth=0.5, alpha=0.5)
for y in y_lines:
    plt.plot([xmin-2*grid_size, xmax+2*grid_size], [y, y], color='gray', linewidth=0.5, alpha=0.5)

plt.show()