#creating reinforcement learning environment
import numpy as np
import gym
from gym.spaces import Box, Discrete, Sequence, MultiBinary, Tuple, MultiDiscrete
from stable_baselines3 import A2C,DQN,PPO

class RlEnv(gym.Env):
    def __init__(self):
        super(RlEnv, self).__init__()
        self.action_space = MultiDiscrete([2, 4, 4])
        self.observation_space = Box(low=-1,high=1,shape=(4,4),dtype=np.int8)

    def reset(self):
        pass

    def step(self, action):
        pass


env = RlEnv()
env.reset()
print(env.action_space.sample())
print(env.observation_space.sample())

#model = PPO('MlpPolicy', env, verbose=1)
#model.learn(total_timesteps = 10)

