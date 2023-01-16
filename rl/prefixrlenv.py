import gym
import numpy as np

class PrefixRlEnv(gym.Env):
    def __init__(self):
        super(PrefixRlEnv, self).__init__()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,4), dtype=np.int8)

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass


env = PrefixRlEnv()
obs = [[1, -1, -1, -1], [1, 1, -1, -1], [1, 1, 1, -1], [1, 1, 1, 1]]

print(env.observation_space.sample())
print(env.observation_space.contains(obs))
