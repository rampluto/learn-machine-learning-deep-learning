import optuna 
from env import CartesianEnv
import gym
import math
import csv
import os
from gym.spaces import Box, Discrete
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from typing import Any
from typing import Dict
import torch
import torch.nn as nn

def train(kwargs):
    env = CartesianEnv()
    obs = env.reset()

    #train agent
    print("************* starting parallel training ************")

    env = make_vec_env(CartesianEnv, n_envs = 5, vec_env_cls = SubprocVecEnv)

    model = A2C('MlpPolicy', env, tensorboard_log = "./training_logs/", **kwargs)
    model.learn(total_timesteps = 100000, tb_log_name="ppo_first_run")
    model.save("models/ppo_first_run")

def evaluate():
    env = CartesianEnv()
    model = PPO.load("./models/ppo_first_run")

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, state_ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward+=reward
        print("reward is ",reward)
        print("obs is ", obs)
    
    return total_reward



def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "verbose": 1,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }

def objective(trial):
    kwargs = sample_a2c_params(trial)
    train(kwargs)
    reward = evaluate()
    return reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    try:
        study.optimize(objective, n_trials=10)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
    



