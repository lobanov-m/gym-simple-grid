import gym
import gym_minigrid


env = gym.make('MiniGrid-Empty-5x5-v0')
action_space = env.action_space
action_space.sample()

for i in range(100):
    env.render()
    observation, reward, done, info = env.step(action_space.sample())
    print(f'\rStep: {i}; Reward: {reward}')
    if done:
        break
