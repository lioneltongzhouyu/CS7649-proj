import argparse
from ast import arg
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
# additional import for gail
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import seals
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C
import os
import pickle


from FolderA import FolderB

'''
Parse
'''
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, help="0: leg-0.2, 1: leg-0.3, 10: default")
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda if True, else CPU")
args = parser.parse_args()
model_type = args.model


'''
Create env
'''

if model_type == 0:
    env_name = 'AntLeg2-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_2'
elif model_type == 1:
    env_name = 'AntLeg3-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_3'
elif model_type == 10:
    env_name = "seals/Ant-v0"
    save_path = 'checkpoints_experts/ant_policy_leg_default'
    

if not os.path.exists(save_path):
    os.makedirs(save_path)

env = gym.make(env_name)
device = 'cuda' if args.cuda else 'cpu'


'''
EXPERT
'''

expert = SAC("MlpPolicy", env, verbose=1, device=device)
expert.learn(2e6, n_eval_episodes=20)  # Note: set to 100000 to train a proficient expert

# Save expert
expert.save(save_path)

# expert reward
reward, _ = evaluate_policy(expert, env, 100, return_episode_rewards=True)
print("Expert reward: ", np.mean(reward))


'''
Collect Rollouts
'''

env.reset()

expert = SAC.load(save_path, device=device)
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
)
transitions = rollout.flatten_trajectories(rollouts)

with open(os.path.join(save_path, 'transition.pkl'), 'wb') as outp:
    pickle.dump(transitions, outp, pickle.HIGHEST_PROTOCOL)
