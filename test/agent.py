import gym
import rsoccer_gym
import warnings

warnings.filterwarnings("ignore") # ignoring warnings

# Using VSS Single Agent env
env = gym.make('VSS-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        env.render()
    print(reward)