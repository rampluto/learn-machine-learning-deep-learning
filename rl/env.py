#creating reinforcement learning environment
import numpy as np
import gym
import random
from gym.spaces import Box, Discrete, Sequence, MultiBinary, Tuple, MultiDiscrete
from stable_baselines3 import A2C,DQN,PPO

class RlEnv(gym.Env):
    def __init__(self):
        super(RlEnv, self).__init__()
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]),high=np.array([100]))
        self.state = 38+random.randint(-3,3)
        self.shower_length = 60

    def reset(self):
        self.state = 38+random.randint(-3,3)
        self.shower_length = 60
        return self.state 

    def step(self, action):
        self.state+=action-1
        self.shower_length-=1

        if self.state >=37 and self.state <=39:
            reward = 1
        else:
            reward = -1

        # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        pass 


env = RlEnv()
env.reset()
print(env.action_space.n)
print(env.observation_space.shape)

#model = A2C("MlpPolicy", env).learn(total_timesteps=1000)

