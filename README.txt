README

Fast Tagger V1 is a project with a runner and a tagger in a 10x10 grid, where the runner is an RL agent implementing learning algorithms to stray away from the tagger.

grid_world_env.py - shows the environment without any RL algorithms used.

grid_world_vi.py - shows the environment with the agent using Value Iteration as a learning algorithm.

grid_world_vi_no_stag.py - the same environment but with penalties for stagnation

grid_world_vi_smart_tag.py - the same environment but with a smart tagger

grid_world_vi_better_tag.py - the same environment but with a bit of randomness for the tagger, to help the agent learn better

grid_world_td_sarsa.py - the environment where the agent implements Temporal Difference algorithm with an on-policy control algorithm called SARSA

CONTACT
saipranithbhagavatula@gmail.com
