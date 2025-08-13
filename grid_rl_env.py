import copy
import random


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
                if grid_to_search[i][j] == 4:
                    return (i, j)
        return None

    def reset(self):
        self.grid = copy.deepcopy(self.original_grid)
        self.pos = self.start
        self.has_package = False
        self.packages_left = sum(row.count(2) for row in self.grid)
        self.delivery_log = []
        self.last_pickup = None
        return (self.pos, self.has_package, self.packages_left)

    def get_actions(self, pos):
        actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.grid[nx][ny] != 1:
                    actions.append((nx, ny))
        return actions

    def step(self, action):
        self.pos = action
        cell_value = self.grid[action[0]][action[1]]
        reward = -1  # step cost
        done = False

        # Pick up a package
        if not self.has_package and cell_value == 2:
            reward += 50
            self.has_package = True
            self.last_pickup = action
            self.grid[action[0]][action[1]] = 0

        # Deliver to shelf
        elif self.has_package and cell_value == 3:
            reward += 200
            self.has_package = False
            self.delivery_log.append(
                f"Delivered package from {self.last_pickup} to {action}"
            )
            self.last_pickup = None
            self.grid[action[0]][action[1]] = 0
            self.packages_left -= 1
            if self.packages_left == 0:
                reward += 10000
                done = True

        # Passing over another package while carrying one → penalty
        elif self.has_package and cell_value == 2:
            reward -= 5

        return (self.pos, self.has_package, self.packages_left), reward, done


def train_agent(env,
                episodes=800,
                alpha=0.1,
                gamma=0.95,
                epsilon_start=0.5,
                epsilon_min=0.05,
                epsilon_decay=0.995,
                max_steps=500):

    q_table = {}

    def get_q(state, action):
        return q_table.get((state, action), 0.0)

    epsilon = epsilon_start
    for _ in range(episodes):
        state = env.reset()
        done = False
        prev_pos = None

        for _ in range(max_steps):
            actions = env.get_actions(state[0])
            if not actions:
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                qs = [get_q(state, a) for a in actions]
                max_q = max(qs)
                best_actions = [a for a, q in zip(actions, qs) if q == max_q]
                action = random.choice(best_actions)

            # Discourage immediate backtracking
            if prev_pos and action == prev_pos and len(actions) > 1:
                actions.remove(action)
                action = random.choice(actions)

            next_state, reward, done = env.step(action)

            next_actions = env.get_actions(next_state[0])
            max_future_q = max((get_q(next_state, a) for a in next_actions), default=0.0)

            old_q = get_q(state, action)
            q_table[(state, action)] = old_q + alpha * (reward + gamma * max_future_q - old_q)

            state = next_state
            prev_pos = state[0]

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # After training, reconstruct path
    path, delivery_log = reconstruct_path(env, q_table)
    print("\nFinal Path:", path)
    print("\nDelivery Log:")
    for log in delivery_log:
        print(" -", log)
    return path, q_table, delivery_log


def reconstruct_path(env, q_table, max_steps=2000):
    state = env.reset()
    path = [env.start]
    steps = 0

    while steps < max_steps:
        actions = env.get_actions(state[0])
        if not actions:
            break

        qs = [q_table.get((state, a), -float('inf')) for a in actions]
        max_q = max(qs)
        best_actions = [a for a, q in zip(actions, qs) if q == max_q]
        action = random.choice(best_actions)

        state, _, done = env.step(action)
        path.append(action)
        steps += 1

        if done:
            break

    return path, env.delivery_log
