import os
from ipywidgets import interact
from IPython.display import display, Image, clear_output
from IPython import embed
import gym_gmazes
import jax
import torch
import numpy as np
import brax
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import SAC, TD3
from xpag.tools import learn

print(jax.lib.xla_bridge.get_backend().platform)

num_envs = 1  # the number of rollouts in parallel during training
env, eval_env, env_info = gym_vec_env('GMazeDCILDubins-v0', num_envs)

agent = SAC(
    env_info['observation_dim'] if not env_info['is_goalenv']
    else env_info['observation_dim'] + env_info['desired_goal_dim'],
    env_info['action_dim'],
    {}
)
sampler = DefaultEpisodicSampler() if not env_info['is_goalenv'] else HER(env.compute_reward)
buffer = DefaultEpisodicBuffer(
    max_episode_steps=env_info['max_episode_steps'],
    buffer_size=1_000_000,
    sampler=sampler
)
goalsetter = DefaultGoalSetter()

batch_size = 256
gd_steps_per_step = 1
# start_training_after_x_steps = env_info['max_episode_steps'] * 10
start_training_after_x_steps = 5_000
max_steps = 10_000_000
evaluate_every_x_steps = 1_000
save_agent_every_x_steps = 100_000
save_dir = os.path.join(os.path.expanduser('~'), 'results', 'xpag', 'dcil')
save_episode = True
# plot_projection = True
def plot_projection(x):
    return x[0:2]


@torch.no_grad()
def goal_d(goal_a, goal_b):
    # assert goal_a.shape == goal_b.shape
    if torch.is_tensor(goal_a):
        return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
    else:
        return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)


@torch.no_grad()
def def_compute_reward(
        achieved_goal,
        desired_goal,
        info
):
    distance_threshold = 0.1
    reward_type = "sparse"
    d = goal_d(achieved_goal, desired_goal)
    if reward_type == "sparse":
        # if torch.is_tensor(achieved_goal):
        #     return (d < distance_threshold).double()
        # else:
        return 1.0 * (d < distance_threshold)
    else:
        return -d

env.set_reward_function(def_compute_reward)
eval_env.set_reward_function(def_compute_reward)

env.reset_done = env.reset_DCIL
eval_env.reset_done = eval_env.reset_DCIL


def gsreset(envir, observation):
    # print('gsreset')
    newgoal = np.random.random(observation['desired_goal'].shape) * 2
    observation['desired_goal'] = newgoal
    envir.unwrapped.goal = torch.tensor(newgoal).to(env.device)
    # embed()
    return observation


# def gsstep(envir, observation, reward, done, info):
#     # print('gsstep')
#     # observation['desired_goal'] = last_saved_goal
#     # envir.goal = torch.tensor(last_saved_goal).to(env.device)
#     return observation, reward, done, info


goalsetter.reset = gsreset
goalsetter.reset_done = gsreset
# goalsetter.step = gsstep

embed()

learn(
    env,
    eval_env,
    env_info,
    agent,
    buffer,
    goalsetter,
    batch_size,
    gd_steps_per_step,
    start_training_after_x_steps,
    max_steps,
    evaluate_every_x_steps,
    save_agent_every_x_steps,
    save_dir,
    save_episode,
    plot_projection,
)

