import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    for episode in range(no_episodes):
        state, _ = env.reset()
        state = tuple(state)
        total_reward = 0

        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = tuple(next_state)
            total_reward += reward

            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
    np.save(q_table_save_path, q_table)
    print("Training completed and Q-table saved.")

def visualize_q_table(goal_coordinates=(3, 2),
                      wall_positions=[[1, 2], [2, 2], [3, 1]],
                      danger_zone=[0, 4],
                      actions=["Up", "Down", "Left", "Right"],
                      q_values_path="q_table.npy"):

    try:
        q_table = np.load(q_values_path)
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            mask[goal_coordinates[0], goal_coordinates[1]] = True
            mask[danger_zone[0], danger_zone[1]] = True
            for wall in wall_positions:
                mask[wall[0], wall[1]] = True

            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            ax.text(danger_zone[1] + 0.5, danger_zone[0] + 0.5, 'D', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

            for wall in wall_positions:
                ax.text(wall[1] + 0.5, wall[0] + 0.5, 'W', color='black',
                        ha='center', va='center', weight='bold', fontsize=14)

            ax.set_title(f"Action: {action}")

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Q-table file not found. Please train the agent first.")
