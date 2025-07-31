from AD_env import create_env
from Q_learning import train_q_learning, visualize_q_table

# Flags
train = True
visualize_results = True

# Hyperparameters
learning_rate = 0.1
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
no_episodes = 500

# Environment static details
goal_coordinates = (3, 2)
wall_positions = [[1, 2], [2, 2], [3, 1]]
danger_zone = [0, 4]

if train:
    env = create_env()

    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    visualize_q_table(goal_coordinates=goal_coordinates,
                      wall_positions=wall_positions,
                      danger_zone=danger_zone,
                      q_values_path="q_table.npy")
