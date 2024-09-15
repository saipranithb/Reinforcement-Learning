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

        self.gamma = 0.9
        self.theta = 1e-4
        self.reset()

    def reset(self):
        self.runner_pos = [0, 0]
        self.tagger_pos = [9, 9]
        self.explosion_var = [0, 0]
        self.total_reward = 0
        self.time_remain = 100
        self.termin_var = False
        self.obs = (self.runner_pos[0], self.runner_pos[1], self.tagger_pos[0], self.tagger_pos[1])

        self.V = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))  # Value table V(s)

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

    def value_iteration(self):
        # Perform Value Iteration on the grid world

        delta = float('inf')  # Track the max change in value
        while delta > self.theta:  # Until convergence
            delta = 0  # Reset delta for this iteration

            # Iterate over all states in the grid world
            for runner_x in range(self.grid_size):
                for runner_y in range(self.grid_size):
                    for tagger_x in range(self.grid_size):
                        for tagger_y in range(self.grid_size):
                            # Get the current state
                            state = (runner_x, runner_y, tagger_x, tagger_y)

                            # Skip terminal states (when the runner is caught)
                            if state[:2] == state[2:]:
                                continue

                            old_value = self.V[state]  # Current value of the state

                            # Find the best action using the Bellman equation
                            best_value = float('-inf')
                            for action in range(self.action_space.n):
                                next_state, reward = self._simulate_step(state, action)
                                value = reward + self.gamma * self.V[next_state]
                                best_value = max(best_value, value)

                            # Update the value for the current state
                            self.V[state] = best_value

                            # Calculate the change in value (delta)
                            delta = max(delta, abs(old_value - self.V[state]))

        print("Value Iteration Converged!")

    def _simulate_step(self, state, action):
        runner_x, runner_y, tagger_x, tagger_y = state

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
        next_runner_x = min(max(runner_x + move[0], 0), self.grid_size - 1)
        next_runner_y = min(max(runner_y + move[1], 0), self.grid_size - 1)

        next_tagger_x, next_tagger_y = self._get_tagger_move_simulation(tagger_x, tagger_y, next_runner_x, next_runner_y)

        if (next_runner_x, next_runner_y) == (next_tagger_x, next_tagger_y):
            reward = -100  # Caught
        else:
            reward = 1  # Survival

        next_state = (next_runner_x, next_runner_y, next_tagger_x, next_tagger_y)
        return next_state, reward

    def visualize_value_function(self):
        runner_value_grid = np.zeros((self.grid_size, self.grid_size))

        for runner_x in range(self.grid_size):
            for runner_y in range(self.grid_size):
                total_value = 0
                count = 0
                for tagger_x in range(self.grid_size):
                    for tagger_y in range(self.grid_size):
                        state = (runner_x, runner_y, tagger_x, tagger_y)
                        total_value += self.V[state]
                        count += 1

                runner_value_grid[runner_x, runner_y] = total_value / count

        # Plotting the value function as a heatmap
        plt.imshow(runner_value_grid, cmap='coolwarm', origin='upper')
        plt.colorbar(label='Value')
        plt.title("Runner's State Value Function")
        plt.xlabel('Y Position')
        plt.ylabel('X Position')
        plt.show()
    
    def _get_tagger_move(self):

        dx = self.runner_pos[0] - self.tagger_pos[0]
        dy = self.runner_pos[1] - self.tagger_pos[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)

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

    def _get_tagger_move_simulation(self, tagger_x, tagger_y, runner_x, runner_y):
        dx = runner_x - tagger_x
        dy = runner_y - tagger_y
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx > abs_dy:
            if dx > 0:
                return min(tagger_x + 2, self.grid_size - 1), tagger_y  # Move right
            else:
                return max(tagger_x - 2, 0), tagger_y  # Move left
        else:
            if dy > 0:
                return tagger_x, min(tagger_y + 2, self.grid_size - 1)  # Move down
            else:
                return tagger_x, max(tagger_y - 2, 0)  # Move up
            
    def render(self):

        if self.termin_var:
            plt.clf()

            plt.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='Greys', vmin=0, vmax=1) # Grid Layout

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


        plt.clf()

        plt.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='Greys', vmin=0, vmax=1)

        for x in range(self.grid_size):
            plt.axhline(y=x - 0.5, color='black', linestyle='--', linewidth=0.5)
            plt.axvline(x=x - 0.5, color='black', linestyle='--', linewidth=0.5)
        

        plt.scatter(self.runner_pos[1], self.runner_pos[0], color='blue', s=100, label='Runner')
        plt.scatter(self.tagger_pos[1], self.tagger_pos[0], color='red', s=100, label='Tagger')


        plt.legend(loc='upper right')
        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(self.grid_size - 0.5, -0.5)
        plt.gca().set_aspect('equal', adjustable='box')

        # Display the details on sidebar
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 3, f"Remaining Steps: {self.time_remain}", fontsize=10, verticalalignment='center', color='blue')
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 2, "Survival Reward: +1", fontsize=12, verticalalignment='center')
        plt.text(self.grid_size + 0.5, self.grid_size // 2 - 1, "Contact Reward: -100", fontsize=12, verticalalignment='center')
        plt.text(self.grid_size + 0.5, self.grid_size // 2, f"Total Reward: {self.total_reward}", fontsize=12, verticalalignment='center', color='red')

        plt.pause(0.2)  # Pause for a teeny bit

if __name__ == "__main__":
    env = GridWorldEnv()
    observation, info = env.reset()
    
    env.value_iteration()
    env.visualize_value_function()
    
    num_timesteps = 100
    plt.ion()

    for i in range(num_timesteps):
        runner_x, runner_y, tagger_x, tagger_y = observation

        best_action = None
        best_value = float('-inf')
        for action in range(env.action_space.n):
            next_state, reward = env._simulate_step(observation, action)
            value = reward + env.gamma * env.V[next_state]
            if value > best_value:
                best_value = value
                best_action = action

        observation, reward, terminated, truncated, info = env.step(best_action)
        
        env.render()
        
        if terminated:
            observation, info = env.reset()

    plt.ioff()
    plt.show()
    env.close()
