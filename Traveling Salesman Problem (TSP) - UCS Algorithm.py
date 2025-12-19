# =====================================================
# IMPORT REQUIRED LIBRARIES
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
import heapq
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
    def plot_route(self, path, title, cost, runtime, iterations):
        x = self.cities[:, 0]
        y = self.cities[:, 1]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y)

        for i in range(self.num_cities):
            plt.text(x[i] + 0.01, y[i] + 0.01, str(i))

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            plt.plot([x[a], x[b]], [y[a], y[b]])

        info = (
            f"Cost: {cost:.2f}\n"
            f"Runtime: {runtime:.4f}s\n"
            f"Iterations: {iterations}"
        )

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
# UNIFORM COST SEARCH (UCS) — PROVABLY OPTIMAL
# =====================================================
def uniform_cost_search_tsp(env):

    start = 0
    start_state = (start, frozenset([start]))

    frontier = [(0.0, start_state, [start])]
    explored = {}
    iterations = 0

    while frontier:
        iterations += 1
        cost, (current, visited), path = heapq.heappop(frontier)

        state = (current, visited)
        if state in explored and explored[state] <= cost:
            continue
        explored[state] = cost

        # ---------------------------------------------
        # GOAL STATE: ALL CITIES VISITED AND BACK AT START
        # ---------------------------------------------
        if current == start and len(visited) == env.num_cities:
            return path, cost, iterations

        # ---------------------------------------------
        # EXPAND SUCCESSORS
        # ---------------------------------------------
        for next_city in range(env.num_cities):

            if next_city not in visited:
                new_visited = visited | {next_city}
                new_cost = cost + env.distance_matrix[current][next_city]

                heapq.heappush(
                    frontier,
                    (new_cost, (next_city, new_visited), path + [next_city])
                )

            elif next_city == start and len(visited) == env.num_cities:
                new_cost = cost + env.distance_matrix[current][start]

                heapq.heappush(
                    frontier,
                    (new_cost, (start, visited), path + [start])
                )

    return None, float("inf"), iterations


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    env = TSPEnvironment(num_cities=15)

    start_time = time.time()
    path, cost, iterations = uniform_cost_search_tsp(env)
    runtime = time.time() - start_time

    env.plot_route(
        path,
        "Uniform Cost Search (Optimal)",
        cost,
        runtime,
        iterations
    )
