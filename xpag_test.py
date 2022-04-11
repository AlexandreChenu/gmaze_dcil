import os
from ipywidgets import interact
from IPython.display import display, Image, clear_output
from IPython import embed
import gym_gmazes
import jax
import brax
from xpag.wrappers import gym_vec_env
from xpag.buffers import DefaultEpisodicBuffer
from xpag.samplers import DefaultEpisodicSampler, HER
from xpag.goalsetters import DefaultGoalSetter
from xpag.agents import SAC
from xpag.tools import learn

print(jax.lib.xla_bridge.get_backend().platform)

num_envs = 10  # the number of rollouts in parallel during training
env, eval_env, env_info = gym_vec_env('GMazeDCILDubins-v0', num_envs)

embed()