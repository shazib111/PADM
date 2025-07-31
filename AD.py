import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random

class ADEnv(gym.Env):
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1, 1])
        self.goal_state = np.array([3, 2])
        self.wall_positions = [[1, 2], [2, 2], [3, 1]]
        self.danger_zone = [0, 4]

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        self.fig, self.ax = plt.subplots()
        plt.show(block=False)
        self.last_event = ""
        self.total_reward = 0

    def reset(self):
        
        # Randomly initialize the agent's position
        self.agent_state = np.array([random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)])

        # If the agent starts in the danger zone, give a penalty but don't end the episode
        if list(self.agent_state) == self.danger_zone:
            self.last_event = "Entered Danger Zone!"
            reward = -1  # Smaller penalty, not an immediate termination
            done = False  # Do not end the episode yet, let the agent move first
        else:
            self.last_event = "Moving..."
            reward = 0
            done = False

        self.total_reward = reward
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}
        
        return self.agent_state, info

    def step(self, action):
        row, col = self.agent_state
        original_pos = self.agent_state.copy()

        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and row < self.grid_size - 1:
            row += 1
        elif action == 2 and col > 0:
            col -= 1
        elif action == 3 and col < self.grid_size - 1:
            col += 1

        new_pos = np.array([row, col])

        if list(new_pos) in self.wall_positions:
            self.last_event = "Hit_wall"
            reward = -2
            done = False
            info = {"event": self.last_event}
            return original_pos, reward, done, info

        if list(new_pos) == self.danger_zone:
            self.agent_state = new_pos
            self.last_event = "Entered Danger Zone!"
            reward = -3
            done = True
            info = {"event": self.last_event}
            return self.agent_state, reward, done, info

        self.agent_state = new_pos
        if np.array_equal(self.agent_state, self.goal_state):
            reward = 15
            done = True
            self.last_event = "Reached_goal"
        else:
            reward = -1
            done = False
            self.last_event = "Moving..."

        self.total_reward += reward
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"event": self.last_event, "Distance to Goal": distance_to_goal}
        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True)

        self.ax.plot(self.goal_state[1], self.goal_state[0], "go", markersize=15, label="Goal")
        for wall in self.wall_positions:
            self.ax.add_patch(plt.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1, color="black"))
        dz = self.danger_zone
        self.ax.add_patch(plt.Rectangle((dz[1] - 0.5, dz[0] - 0.5), 1, 1, color="red", alpha=0.5))
        self.ax.plot(self.agent_state[1], self.agent_state[0], "bo", markersize=12, label="Agent")

        color_map = {
            "Reached_goal": "green",
            "Entered Danger Zone!": "red",
            "Hit_wall": "orange",
            "Moving...": "blue"
        }
        color = color_map.get(self.last_event, "black")
        self.ax.text(0, -1, f"{self.last_event}", fontsize=12, color=color, fontweight="bold")

        self.ax.invert_yaxis()
        self.ax.legend()
        plt.pause(0.1)

    def close(self):
        plt.close()

def create_env():
    return ADEnv()
