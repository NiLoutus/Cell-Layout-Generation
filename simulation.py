import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

def intensity_map_generation(points, grid_size = 50, layout_size = 10000):
	num_grids = layout_size // grid_size
	intensity_map = np.zeros((num_grids, num_grids))

	for point in points:
	    # print(point)
	    grid_x = int(point[0] // grid_size)
	    grid_y = int(point[1] // grid_size)

	    intensity_map[grid_x, grid_y] += 1

	# Normalize by grid area to get intensity
	intensity_map /= grid_size**2

	# Visualize the intensity map
	plt.imshow(intensity_map.T, cmap='hot', interpolation='nearest', origin='lower')
	plt.colorbar()
	plt.title("Intensity Map")
	plt.show()

	return intensity_map


def show_scatter_points(points):
	plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.title("Scatter Plot of Points")
	plt.show()


def simulate_points_in_grid(intensity_map, grid_size):
    simulated_points = []

    # Loop through each grid
    for i in range(intensity_map.shape[0]):
        for j in range(intensity_map.shape[1]):
            # Number of points in this grid
            num_points = np.random.poisson(intensity_map[i, j] * grid_size**2)

            # Generate random points within this grid
            points = np.random.rand(num_points, 2)
            points[:, 0] = points[:, 0] * grid_size + i * grid_size
            points[:, 1] = points[:, 1] * grid_size + j * grid_size

            simulated_points.append(points)

    return np.vstack(simulated_points)

def simulate_points_in_grid_hardcore(intensity_map, grid_size, min_distance):
    simulated_points = []

    # Loop through each grid
    for i in range(intensity_map.shape[0]):
        for j in range(intensity_map.shape[1]):
            # Number of points in this grid
            num_points = np.random.poisson(intensity_map[i, j] * grid_size**2)
            points = []
            attempts = 0
            max_attempts = num_points * 100  # Limit attempts to avoid infinite loops

            while len(points) < num_points and attempts < max_attempts:
                new_point = np.random.rand(2) * grid_size
                new_point[0] += i * grid_size
                new_point[1] += j * grid_size

                # Check minimum distance with all existing points in the grid
                if all(np.linalg.norm(new_point - point) >= min_distance for point in points):
                    points.append(new_point)
                attempts += 1

            # Append the points from this grid to the list of all points
            if points:
                simulated_points.extend(points)

    return np.array(simulated_points)

