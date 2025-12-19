# =====================================================
# IMPORT REQUIRED LIBRARIES
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
import time


# =====================================================
# TSP ENVIRONMENT CLASS
# =====================================================
class TSPEnvironment:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.reset()

    # =================================================
    # DISTANCE MATRIX CALCULATION
    # =================================================
    def _compute_distances(self):
        coords = self.cities
        return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    # =================================================
    # RESET ENVIRONMENT (RANDOM CITY GENERATION)
    # =================================================
    def reset(self):
        self.cities = np.random.rand(self.num_cities, 2)
        self.distance_matrix = self._compute_distances()

    # =================================================
    # VISUALIZATION FUNCTION
    # =================================================
    def plot_route(self, path, title, cost, runtime):
        x = self.cities[:, 0]
        y = self.cities[:, 1]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y)

        for i in range(self.num_cities):
            plt.text(x[i] + 0.01, y[i] + 0.01, str(i))

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            plt.plot([x[a], x[b]], [y[a], y[b]])

        info = f"Cost: {cost:.2f}\nRuntime: {runtime:.4f}s"

        plt.gca().text(
            0.02, 0.95, info,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.85)
        )

        plt.title(title)
        plt.grid(True)
        plt.show()


# =====================================================
# GREEDY / NEAREST NEIGHBOR ALGORITHM
# =====================================================
def greedy_tsp(env):
    """
    Greedy algorithm: always go to the nearest unvisited city
    """

    current = 0
    visited = [current]
    unvisited = set(range(env.num_cities)) - {current}

    while unvisited:
        next_city = min(unvisited,
                        key=lambda c: env.distance_matrix[current][c])
        visited.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    visited.append(visited[0])  # return to start

    # Compute total cost
    cost = sum(env.distance_matrix[visited[i]][visited[i + 1]]
               for i in range(len(visited) - 1))

    return visited, cost


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    env = TSPEnvironment(num_cities=15)

    # -------------------------------
    # RUN GREEDY
    # -------------------------------
    start_time = time.time()
    path, cost = greedy_tsp(env)
    runtime = time.time() - start_time

    # -------------------------------
    # PLOT RESULT
    # -------------------------------
    env.plot_route(
        path,
        "Greedy / Nearest Neighbor",
        cost,
        runtime
    )