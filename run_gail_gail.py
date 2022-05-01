import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
# additional import for gail

from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from FolderA import FolderB
import gym
import seals
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C

import torch
import pickle
import os

from imitation.algorithms.adversarial.gail import GAIL

# from stable_baselines import GAIL

'''
Parse
'''
parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_model", type=int, help="0: leg-0.2, 1: leg-0.3, 10: default")
# parser.add_argument("--train_model", type=int, help="0: leg-0.2, 1: leg-0.3, 10: default")
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda if True, else CPU")
parser.add_argument("--seed", type=int, default=True, help="seed")
args = parser.parse_args()

pretrain_model_type = args.pretrain_model
# train_model_type = args.train_model
seed = str(args.seed)


my_log_dir = "output/"

gail_epoch = 30

'''
Create env
'''

if pretrain_model_type == 0:
    env_name = 'AntLeg2-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_2/rollouts.pkl'
elif pretrain_model_type == 1:
    env_name = 'AntLeg3-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_3/rollouts.pkl'
elif pretrain_model_type == 10:
    # env_name = "seals/Ant-v0"
    env_name = "Antorigin-v0"
    save_path = 'checkpoints_experts/ant_policy_leg_default/rollouts.pkl'


my_log_dir = os.path.join("output/gail_gail/", env_name, seed)

if not os.path.exists(my_log_dir):
    os.makedirs(my_log_dir)

if not os.path.exists(save_path):
    print("[ERROR] Pretrained expert model not exist!")

env = gym.make(env_name)
device = 'cuda' if args.cuda else 'cpu'

with open(save_path, 'rb') as inp:
    rollouts = pickle.load(inp)


transitions = rollout.flatten_trajectories(rollouts)



'''
Pretrain with one GAIL
'''

env.reset()

'''
Train with GAIL by using pretrained BC
'''

venv = DummyVecEnv([lambda: env] * 8)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    device=device,
)


reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True
)


gail_trainer.train(gail_epoch * 10000)  # Note: set to 300000 for better results





policy_trained_by_gail = gail_trainer.policy

print("====================================================================================")

env.reset()

'''
Train with GAIL by using pretrained BC
'''
env_name = "Antorigin-v0"

venv = DummyVecEnv([lambda: gym.make(env_name)] * 8)

learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    device=device,
)

# Replace policy with pretrained BC policy
learner.policy = policy_trained_by_gail

reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    
    init_tensorboard=True,
    init_tensorboard_graph=True,
    log_dir=my_log_dir,
    allow_variable_horizon=True
)

# Actual training
gail_trainer.train(gail_epoch * 10000)  # Note: set to 300000 for better results

