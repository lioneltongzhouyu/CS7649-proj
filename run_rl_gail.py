from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.ppo import MlpPolicy
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
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
from gym.spaces import Box, Discrete


import argparse
import os
import gym
import seals
import matplotlib.pyplot as plt
import numpy as np

from FolderA import FolderB
import pickle

'''
EXPERT
'''
#Humanoid, HalfCheetah, Hopper, Ant, Walker2d

# env_name = 'AntLeg2-v0'
# save_path = 'checkpoints_experts/ant_policy_leg_2/rollouts.pkl'

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda if True, else CPU")
parser.add_argument("--seed", type=int, default=True, help="seed")
args = parser.parse_args()

# train_model_type = args.train_model
seed = str(args.seed)

# env_name = "seals/Ant-v0"
env_name = "Antorigin-v0"
save_path = 'checkpoints_experts/ant_policy_leg_default/rollouts.pkl'
device = 'cuda'

env = gym.make(env_name)
# expert = SAC.load(save_path, device=device)

with open(save_path, 'rb') as inp:
    rollouts = pickle.load(inp)


transitions = rollout.flatten_trajectories(rollouts)
'''
NOISE
'''
o_size = env.observation_space.low.shape[0]
a_size = env.action_space.low.shape[0]
env.observation_space.low += np.random.normal(0,1,o_size)
env.observation_space.high += np.random.normal(0,1,o_size)
env.action_space.low += np.random.normal(0,1,a_size)
env.action_space.high += np.random.normal(0,1,a_size)

'''
PRETRAIN W RL
'''
rl_model = A2C('MlpPolicy', env, device=device,)
reward_before_training, _ = evaluate_policy(rl_model.policy, env, 10)
print(f"Reward before training RL: {reward_before_training}")

rl_model.learn(100000) 

# Reward after training
reward_after_training, _ = evaluate_policy(rl_model.policy, env, 10)
print(f"Reward after training RL: {reward_after_training}")

# Get trained policy
policy_trained_by_a2c = rl_model.policy









'''
GAIL
'''
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
# Replace policy with pretrained RL policy
learner.policy = policy_trained_by_a2c

reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)


my_log_dir = os.path.join("output/rl_gail/", env_name, seed)

if not os.path.exists(my_log_dir):
    os.makedirs(my_log_dir)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    init_tensorboard=True,
    init_tensorboard_graph=True,
    log_dir=my_log_dir,
)

# Actual training
gail_trainer.train(300000)




