import copy
import random


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GridEnv:
    def __init__(self, grid):
        self.original_grid = copy.deepcopy(grid)
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = self.find_robot_start(grid) or (0, 0)
        self.reset()

    def find_robot_start(self, grid_to_search):
        for i in range(self.rows):
            for j in range(self.cols):
                if grid_to_search[i][j] == 4:  # Robot start
                    return (i, j)
        return None

    # ---------- Battery helpers ----------
    def _charge_level(self):
        if self.charge > 66:
            return "HIGH"
        elif self.charge > 33:
            return "MED"
        else:
            return "LOW"

    # ---------- Env API ----------
    def reset(self):
        self.grid = copy.deepcopy(self.original_grid)
        self.pos = self.start
        self.has_package = False
        self.packages_left = sum(row.count(2) for row in self.grid)
        self.delivery_log = []
        self.last_pickup = None
        self.pickup_pos = None
        self.steps_since_pickup = 0

        # Battery system (deducted only on delivery)
        self.charge = 100

        return (self.pos, self.has_package, self.packages_left, self._charge_level())

    def positions_with_value(self, value):
        return [(i, j)
                for i in range(self.rows)
                for j in range(self.cols)
                if self.grid[i][j] == value]

    def get_actions(self, pos):
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] != 1:  # not a wall
                    actions.append((nx, ny))
        return actions

    def step(self, action, visited=None, prev_pos=None):
        self.pos = action
        cell_value = self.grid[action[0]][action[1]]

        reward = -50
        done = False

        if cell_value == 6:
            reward -= 20000
            done = True

        # --- Charging station ---
        if cell_value == 5:
            level = self._charge_level()
            if level == "LOW":
                reward += 200
            elif level == "MED":
                reward += 50
            self.charge = 100

        # --- Pick up package ---
        if not self.has_package and cell_value == 2:
            reward += 200
            self.has_package = True
            self.last_pickup = action
            self.pickup_pos = action
            self.steps_since_pickup = 0
            self.grid[action[0]][action[1]] = 0

        # --- Deliver to shelf ---
        elif self.has_package and cell_value == 3:
            reward += 500
            optimal = manhattan(self.pickup_pos, action) if self.pickup_pos else 0
            if self.steps_since_pickup > optimal + 3:
                reward -= 450

            self.charge = max(0, self.charge - self.steps_since_pickup * 3)
            self.has_package = False
            self.delivery_log.append(
                f"Delivered from {self.last_pickup} to {action} "
                f"(steps {self.steps_since_pickup}, optimal {optimal}, charge {self.charge})"
            )
            self.last_pickup = None
            self.pickup_pos = None
            self.steps_since_pickup = 0
            self.grid[action[0]][action[1]] = 0
            self.packages_left -= 1

            if self.packages_left == 0:
                reward += 8000
                done = True

        # Passing over another package while carrying
        elif self.has_package and cell_value == 2:
            reward -= 5

        # Step counter
        if self.has_package:
            self.steps_since_pickup += 1
            if self.steps_since_pickup > 50:  # 🚨 limit
                reward -= 1000
                done = True

        if cell_value == 6:
            reward -= 5000
            done = True
        
        if self.charge <= 0:
            reward -= 500
            done = True

        return (self.pos, self.has_package, self.packages_left, self._charge_level()), reward, done




# ---------- Q-learning ----------
def train_agent(env, episodes=10000, alpha=0.12, gamma=0.96,
                epsilon_start=0.8, epsilon_min=0.05, epsilon_decay=0.970, max_steps=20000):

    q_table = {}  # key: (state, action) -> float

    def get_q(state, action):
        return q_table.get((state, action), 0.0)

    epsilon = epsilon_start

    for _ in range(episodes):
        state = env.reset()
        done = False
        prev_pos = None
        visited = set()

        for _ in range(max_steps):
            actions = env.get_actions(state[0])
            if not actions:
                break

            # ε-greedy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                qs = [get_q(state, a) for a in actions]
                max_q = max(qs)
                best_actions = [a for a, qv in zip(actions, qs) if qv == max_q]
                action = random.choice(best_actions)

            next_state, reward, done = env.step(action, None, prev_pos)
            visited.add(action)

            # Bellman update
            next_actions = env.get_actions(next_state[0])
            max_future_q = max((get_q(next_state, a) for a in next_actions), default=0.0)
            old_q = get_q(state, action)
            q_table[(state, action)] = old_q + alpha * (reward + gamma * max_future_q - old_q)

            prev_pos = state[0]
            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Build greedy rollout
    path, delivery_log = reconstruct_path(env, q_table)
    print("\nFinal Path:", path)
    print("\nDelivery Log:")
    for log in delivery_log:
        print(" -", log)
    return path, q_table, delivery_log


# ---------- Greedy reconstruction ----------
def reconstruct_path(env, q_table, max_steps=2500):
    state = env.reset()
    path = [env.start]
    visited = set([env.start])
    steps = 0

    while steps < max_steps:
        actions = env.get_actions(state[0])
        if not actions:
            break

        qs = [q_table.get((state, a), None) for a in actions]

        if all(v is None for v in qs):
            # unseen state → random move
            unvisited = [a for a in actions if a not in visited]
            action = random.choice(unvisited if unvisited else actions)
        else:
            vals = [(-float('inf') if v is None else v) for v in qs]
            max_q = max(vals)
            best = [a for a, v in zip(actions, vals) if v == max_q]
            unvisited_best = [a for a in best if a not in visited]
            action = random.choice(unvisited_best if unvisited_best else best)

        # Loop breaker: avoid immediate backtrack in reconstruction
        if len(path) > 2 and action == path[-2]:
            alts = [a for a in actions if a != path[-2]]
            if alts:
                action = random.choice(alts)

        state, _, done = env.step(action, visited, path[-1])
        path.append(action)
        visited.add(action)
        steps += 1

        if done:
            break

    return path, env.delivery_log
