from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import seals
import argparse
from FolderA import FolderB



parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_model", type=int, help="0: leg-0.2, 1: leg-0.3, 10: default")
# parser.add_argument("--train_model", type=int, help="0: leg-0.2, 1: leg-0.3, 10: default")
parser.add_argument("--cuda", type=bool, default=True, help="Use cuda if True, else CPU")
parser.add_argument("--seed", type=int, default=True, help="seed")
args = parser.parse_args()
seed = str(args.seed)
pretrain_model_type = args.pretrain_model
if pretrain_model_type == 0:
    env_name = 'AntLeg2-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_2/rollouts.pkl'
elif pretrain_model_type == 1:
    env_name = 'AntLeg3-v0'
    save_path = 'checkpoints_experts/ant_policy_leg_3/rollouts.pkl'
elif pretrain_model_type == 10:
    # env_name = "seals/Ant-v0"
    env_name = "Antorigin-v0"
    save_path = 'checkpoints_experts/ant_policy_leg_default/expert.zip'

env = gym.make(env_name)

env.reset()
device="cpu"
expert = SAC.load(save_path, device=device)
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
)
transitions = rollout.flatten_trajectories(rollouts)

venv = DummyVecEnv([lambda: gym.make(env_name)] * 8)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

my_log_dir = os.path.join("output/basic_gail/", env_name, seed)

if not os.path.exists(my_log_dir):
    os.makedirs(my_log_dir)
    print(my_log_dir)

if not os.path.exists(save_path):
    print("[ERROR] Pretrained expert model not exist!")

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    init_tensorboard = True,
    init_tensorboard_graph = True,
    log_dir=my_log_dir
)
gail_trainer.train(300000)  # Note: set to 300000 for better results
