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
        """
        Initialize the TSP environment
        """
        self.num_cities = num_cities
        self.cities = None
        self.distance_matrix = None
        self.reset()

    # =================================================
    # DISTANCE MATRIX CALCULATION
    # =================================================
    def _compute_distances(self):
        """
        Compute Euclidean distance between every pair of cities
        """
        coords = self.cities
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        return dist

    # =================================================
    # RESET ENVIRONMENT (RANDOM CITY GENERATION)
    # =================================================
    def reset(self):
        """
        Reset environment and randomly generate new cities
        """
        self.cities = np.random.rand(self.num_cities, 2)
        self.distance_matrix = self._compute_distances()

    # =================================================
    # VISUALIZATION FUNCTION FOR SUBPLOTS
    # =================================================
    def plot_route_ax(self, ax, path, title, cost, runtime, iterations=None):
        """
        Plot a route on a given matplotlib axis
        """
        x = self.cities[:, 0]
        y = self.cities[:, 1]

        ax.scatter(x, y)

        for i in range(self.num_cities):
            ax.text(x[i] + 0.01, y[i] + 0.01, str(i), fontsize=9)

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ax.plot([x[a], x[b]], [y[a], y[b]])

        info = f"Cost: {cost:.2f}\nRuntime: {runtime:.4f}s"
        if iterations is not None:
            info += f"\nIterations: {iterations}"

        ax.text(
            0.02, 0.95, info,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.85)
        )

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)


# =====================================================
# HELPER FUNCTION: MINIMUM SPANNING TREE COST
# =====================================================
def mst_cost(env, nodes):
    """
    Compute MST cost using Prim's algorithm
    """
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
# UNIFORM COST SEARCH (UCS) — PROVABLY OPTIMAL
# =====================================================
def uniform_cost_search_tsp(env):
    """
    Uniform Cost Search for TSP
    Correct goal modeling (return edge included)
    """

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
        # GOAL STATE: BACK AT START WITH ALL VISITED
        # ---------------------------------------------
        if current == start and len(visited) == env.num_cities:
            return path, cost, iterations

        # ---------------------------------------------
        # EXPAND SUCCESSORS
        # ---------------------------------------------
        for next_city in range(env.num_cities):

            # Visit unvisited city
            if next_city not in visited:
                new_visited = visited | {next_city}
                new_cost = cost + env.distance_matrix[current][next_city]

                heapq.heappush(
                    frontier,
                    (new_cost, (next_city, new_visited), path + [next_city])
                )

            # Return to start only after all visited
            elif next_city == start and len(visited) == env.num_cities:
                new_cost = cost + env.distance_matrix[current][start]

                heapq.heappush(
                    frontier,
                    (new_cost, (start, visited), path + [start])
                )

    return None, float("inf"), iterations


# =====================================================
# A* HEURISTIC FUNCTION (MST-BASED, ADMISSIBLE)
# =====================================================
def tsp_heuristic(env, current, unvisited):
    """
    MST-based admissible heuristic
    """
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

        if current == start and len(visited) == env.num_cities:
            return path, g, iterations

        for next_city in range(env.num_cities):

            if next_city not in visited:
                new_visited = visited | {next_city}
                g_new = g + env.distance_matrix[current][next_city]
                h_new = tsp_heuristic(
                    env, next_city,
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
# GREEDY / NEAREST NEIGHBOR
# =====================================================
def greedy_tsp(env):

    current = 0
    visited = [current]
    unvisited = set(range(env.num_cities)) - {current}

    while unvisited:
        next_city = min(unvisited,
                        key=lambda c: env.distance_matrix[current][c])
        visited.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    visited.append(visited[0])

    cost = sum(env.distance_matrix[visited[i]][visited[i + 1]]
               for i in range(len(visited) - 1))

    return visited, cost


# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":

    env = TSPEnvironment(num_cities=15)

    # -------------------------------
    # RUN UCS
    # -------------------------------
    t0 = time.time()
    ucs_path, ucs_cost, ucs_iter = uniform_cost_search_tsp(env)
    ucs_time = time.time() - t0

    # -------------------------------
    # RUN A*
    # -------------------------------
    t0 = time.time()
    astar_path, astar_cost, astar_iter = a_star_tsp(env)
    astar_time = time.time() - t0

    # -------------------------------
    # RUN GREEDY
    # -------------------------------
    t0 = time.time()
    greedy_path, greedy_cost = greedy_tsp(env)
    greedy_time = time.time() - t0

    # -------------------------------
    # PLOT RESULTS
    # -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    env.plot_route_ax(
        axes[0], ucs_path, "Uniform Cost Search",
        ucs_cost, ucs_time, ucs_iter
    )

    env.plot_route_ax(
        axes[1], astar_path, "A* Search (MST Heuristic)",
        astar_cost, astar_time, astar_iter
    )

    env.plot_route_ax(
        axes[2], greedy_path, "Greedy Search",
        greedy_cost, greedy_time
    )

    plt.tight_layout()
    plt.show()
