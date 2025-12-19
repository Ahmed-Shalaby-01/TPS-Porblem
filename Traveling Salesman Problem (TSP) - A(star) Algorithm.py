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
# HELPER FUNCTION: MST COST (PRIM'S ALGORITHM)
# =====================================================
def mst_cost(env, nodes):

    if len(nodes) <= 1:
        return 0.0

    visited = {nodes[0]}
    total_cost = 0.0

    while len(visited) < len(nodes):
        best = float("inf")
        next_node = None

        for u in visited:
            for v in nodes:
                if v not in visited:
                    cost = env.distance_matrix[u][v]
                    if cost < best:
                        best = cost
                        next_node = v

        total_cost += best
        visited.add(next_node)

    return total_cost


# =====================================================
# A* HEURISTIC FUNCTION (ADMISSIBLE)
# =====================================================
def tsp_heuristic(env, current, unvisited):

    if not unvisited:
        return env.distance_matrix[current][0]

    unvisited = list(unvisited)

    to_unvisited = min(env.distance_matrix[current][c] for c in unvisited)
    to_start = min(env.distance_matrix[c][0] for c in unvisited)
    mst = mst_cost(env, unvisited)

    return to_unvisited + mst + to_start


# =====================================================
# A* SEARCH ALGORITHM
# =====================================================
def a_star_tsp(env):

    start = 0
    start_state = (start, frozenset([start]))

    frontier = []
    explored = {}
    iterations = 0

    h0 = tsp_heuristic(env, start, set(range(env.num_cities)) - {start})
    heapq.heappush(frontier, (h0, 0.0, start_state, [start]))

    while frontier:
        iterations += 1
        f, g, (current, visited), path = heapq.heappop(frontier)

        state = (current, visited)
        if state in explored and explored[state] <= g:
            continue
        explored[state] = g

        # ---------------------------------------------
        # GOAL STATE: ALL CITIES VISITED AND BACK AT START
        # ---------------------------------------------
        if current == start and len(visited) == env.num_cities:
            return path, g, iterations

        # ---------------------------------------------
        # EXPAND SUCCESSORS
        # ---------------------------------------------
        for next_city in range(env.num_cities):

            if next_city not in visited:
                new_visited = visited | {next_city}
                g_new = g + env.distance_matrix[current][next_city]
                h_new = tsp_heuristic(
                    env,
                    next_city,
                    set(range(env.num_cities)) - new_visited
                )

                heapq.heappush(
                    frontier,
                    (g_new + h_new, g_new,
                     (next_city, new_visited),
                     path + [next_city])
                )

            elif next_city == start and len(visited) == env.num_cities:
                g_new = g + env.distance_matrix[current][start]

                heapq.heappush(
                    frontier,
                    (g_new, g_new,
                     (start, visited),
                     path + [start])
                )

    return None, float("inf"), iterations


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    env = TSPEnvironment(num_cities=15)

    start_time = time.time()
    path, cost, iterations = a_star_tsp(env)
    runtime = time.time() - start_time

    env.plot_route(
        path,
        "A* Search (MST Heuristic)",
        cost,
        runtime,
        iterations

    )
