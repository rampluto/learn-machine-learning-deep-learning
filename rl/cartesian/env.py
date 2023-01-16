import gym
import math
from gym.spaces import Box, Discrete
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C

class CartesianEnv(gym.Env):
    def __init__(self):
        self.observation_space = Box(low=0, high=1, shape=(16,16), dtype='int')
        self.action_space = Discrete(512)
        self.action_meaning = self.action_means()

    def reset(self):
        self.state = np.zeros((16,16), dtype='int')
        self.done = False
        self.episode_length = 5
        self.reward = 0
        return np.array(self.state)

    def step(self, action):
        action = self.action_meaning[int(action)]
        self.episode_length-=1
        info = {}
        m = action[1]
        n = action[2]

        if(action[0]==0):
            self.state[m][n] = 1
            self.reward = self.calculate_reward(m,n)


        else:
            self.state[m][n] = 0
            self.reward = 0

        if(self.episode_length<=0):
            self.done = True
        
        return self.state, self.reward, self.done, info

    def render(self):
        pass

    def action_means(self):
        action_meaning = dict()
        action = 0
        for i in range(16):
            for j in range(16):
                action_meaning[action] = [0, i, j]
                action_meaning[action+1] = [1, i, j]
                action+=2

        return action_meaning

    def calculate_reward(self,m,n):
        return math.dist([m, n], [0 ,0])



if __name__ == '__main__':
    env = CartesianEnv()
    #obs = env.reset()
    #print("initial obs is", obs)
    #print("random obs is", env.observation_space.sample())
    #check_env(env)

    #for _ in range(20):
        #obs = env.reset()
        #done = False
       # while not done:
       #     action = env.action_space.sample()
      #      obs_, reward, done, _ = env.step(action) 
     #       print("obs is",obs_)
    #        print("reward is",reward)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log = "./training_logs/")
    model.learn(total_timesteps = 10000, tb_log_name="ppo_first_run")
    model.save("models/ppo_first_run")

    
