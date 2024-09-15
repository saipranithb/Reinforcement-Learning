import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 10
        self.observation_space = gym.spaces.Tuple((gym.spaces.Discrete(self.grid_size), gym.spaces.Discrete(self.grid_size), gym.spaces.Discrete(self.grid_size), gym.spaces.Discrete(self.grid_size)))
        self.action_space = gym.spaces.Discrete(8)
        self.total_reward = 0
        self.reset()

    def reset(self):
        self.runner_pos = [0, 0]
        self.tagger_pos = [9, 9]
        self.explosion_var = [0, 0]
        self.total_reward = 0
        self.time_remain = 100
        self.termin_var = False
        self.obs = (self.runner_pos[0], self.runner_pos[1], self.tagger_pos[0], self.tagger_pos[1])
        return self.obs, {}

    def step(self, action):
        # Runner
        move_map = {
            0: (0, 1),   # Up
            1: (0, -1),  # Down
            2: (1, 0),   # Right
            3: (-1, 0),  # Left
            4: (1, 1),   # Up-Right
            5: (-1, -1), # Down-Left
            6: (-1, 1),  # Up-Left
            7: (1, -1)   # Down-Right
        }

        move = move_map[action]
        self.runner_pos[0] = min(max(self.runner_pos[0] + move[0], 0), self.grid_size - 1)
        self.runner_pos[1] = min(max(self.runner_pos[1] + move[1], 0), self.grid_size - 1)

        # Tagger
        tagger_move = self._get_tagger_move()
        self.tagger_pos[0] = min(max(self.tagger_pos[0] + tagger_move[0], 0), self.grid_size - 1)
        self.tagger_pos[1] = min(max(self.tagger_pos[1] + tagger_move[1], 0), self.grid_size - 1)

        terminated = self.runner_pos == self.tagger_pos
        reward = -100 if terminated else 1
        self.total_reward += reward
        self.time_remain -= 1

        if terminated:
            self.explosion_var = self.runner_pos
            self.runner_pos = [0, 0]
            self.tagger_pos = [9, 9]
            self.termin_var = True
        else:
            self.termin_var = False
               
        self.obs = (self.runner_pos[0], self.runner_pos[1], self.tagger_pos[0], self.tagger_pos[1])
        return self.obs, reward, False, terminated, {}

    def _get_tagger_move(self):

        dx = self.runner_pos[0] - self.tagger_pos[0]
        dy = self.runner_pos[1] - self.tagger_pos[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Determine the direction (up, down, left, right) to move
        if abs_dx > abs_dy:
            if dx > 0:
                return (2, 0)  # Move right
            else:
                return (-2, 0) # Move left
        else:
            if dy > 0:
                return (0, 2)  # Move down
            else:
                return (0, -2) # Move up

    def render(self):

        if self.termin_var:
            plt.clf()

            plt.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='Greys', vmin=0, vmax=1) # Grid Layout

            # Grid Lines
            for x in range(self.grid_size):
                plt.axhline(y=x - 0.5, color='black', linestyle='--', linewidth=0.5)
                plt.axvline(x=x - 0.5, color='black', linestyle='--', linewidth=0.5)
            
            plt.scatter(self.explosion_var[1], self.explosion_var[0], color='black', s=300, marker='*')
            plt.scatter(self.explosion_var[1], self.explosion_var[0], color='blue', s=100, label='Runner')
            plt.scatter(self.explosion_var[1], self.explosion_var[0], color='red', s=100, label='Tagger')
            
            plt.legend(loc='upper right')
            plt.xlim(-0.5, self.grid_size - 0.5)
            plt.ylim(self.grid_size - 0.5, -0.5)
            plt.gca().set_aspect('equal', adjustable='box')

            # Display the details on sidebar
            plt.text(self.grid_size + 0.5, self.grid_size // 2 - 3, f"Remaining Steps: {self.time_remain}", fontsize=10, verticalalignment='center', color='blue')
            plt.text(self.grid_size + 0.5, self.grid_size // 2 - 2, "Survival Reward: +1", fontsize=12, verticalalignment='center')
            plt.text(self.grid_size + 0.5, self.grid_size // 2 - 1, "Contact Reward: -100", fontsize=12, verticalalignment='center')
            plt.text(self.grid_size + 0.5, self.grid_size // 2, f"Total Reward: {self.total_reward}", fontsize=12, verticalalignment='center', color='red')
            
            plt.pause(0.5)


        plt.clf()  # Clear the previous plot

        # Create a grid background
        plt.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='Greys', vmin=0, vmax=1)

        # Draw grid lines
        for x in range(self.grid_size):
            plt.axhline(y=x - 0.5, color='black', linestyle='--', linewidth=0.5)
            plt.axvline(x=x - 0.5, color='black', linestyle='--', linewidth=0.5)
        

        # Plot runner and tagger
        plt.scatter(self.runner_pos[1], self.runner_pos[0], color='blue', s=100, label='Runner')
        plt.scatter(self.tagger_pos[1], self.tagger_pos[0], color='red', s=100, label='Tagger')


        plt.legend(loc='upper right')
        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(self.grid_size - 0.5, -0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # Display the details on sidebar
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 3, f"Remaining Time: {self.time_remain}", fontsize=10, verticalalignment='center', color='blue')
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 2, "Survival Reward: +1", fontsize=12, verticalalignment='center')
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 1, "Contact Reward: -100", fontsize=12, verticalalignment='center')
        plt.text(self.grid_size + 0.5, self.grid_size // 2, f"Total Reward: {self.total_reward}", fontsize=12, verticalalignment='center', color='red')

        plt.pause(0.2)  # Pause for half a second

if __name__ == "__main__":
    env = GridWorldEnv()

    # policy, V = value_iteration(env)

    observation, info = env.reset()
    num_timesteps = 100
    plt.ion()

    for i in range(num_timesteps):
        
        #state = (observation[0], observation[1], observation[2], observation[3])
        #action = policy[state]
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        env.render()
        if terminated:
            observation, info = env.reset()
            
    plt.ioff()
    plt.show()
    env.close()
