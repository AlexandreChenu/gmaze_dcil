# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
from abc import abstractmethod
from typing import Optional
import gymnasium as gym
from typing import Union
from gymnasium import spaces, utils
from gymnasium import error
import numpy as np
from matplotlib import collections as mc
from IPython import embed

from .maze.maze import Maze

import matplotlib.pyplot as plt


class GoalEnv(gym.Env):
	"""The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

	def reset(self, options=None, seed: Optional[int] = None):
		super().reset(seed=seed)
		# Enforce that each GoalEnv uses a Goal-compatible observation space.
		if not isinstance(self.observation_space, gym.spaces.Dict):
			raise error.Error(
				"GoalEnv requires an observation space of type gym.spaces.Dict"
			)
		for key in ["observation", "achieved_goal", "desired_goal"]:
			if key not in self.observation_space.spaces:
				raise error.Error('GoalEnv requires the "{}" key.'.format(key))

	def compute_reward(self, achieved_goal, desired_goal, info):
		"""Compute the step reward.
		Args:
			achieved_goal (object): the goal that was achieved during execution
			desired_goal (object): the desired goal
			info (dict): an info dictionary with additional information
		Returns:
			float: The reward that corresponds to the provided achieved goal w.r.t. to
			the desired goal. Note that the following should always hold true:
				ob, reward, done, info = env.step()
				assert reward == env.compute_reward(ob['achieved_goal'],
													ob['desired_goal'], info)
		"""
		raise NotImplementedError


def intersect(a, b, c, d):
	x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
	y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
	denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

	t = np.divide(
		((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)),
		denom,
		out=np.zeros_like(denom),
		where=denom != 0,
	)
	criterion1 = np.logical_and(t > 0, t < 1)
	t = np.divide(
		((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)),
		denom,
		out=np.zeros_like(denom),
		where=denom != 0,
	)
	criterion2 = np.logical_and(t > 0, t < 1)

	return np.logical_and(criterion1, criterion2)


class GMazeCommon(Maze):
	def __init__(self, num_envs: int = 1):

		args={}
		args['mazesize'] = 2
		args['random_seed'] = 0
		args['mazestandard'] = False
		args['wallthickness'] = 0.1
		args['wallskill'] = True
		args['targetkills'] = True

		super(GMazeCommon,self).__init__(args['mazesize'],args['mazesize'],seed=args['random_seed'],standard=args['mazestandard'])
		self.maze_size = int(args['mazesize'])

		self.num_envs = num_envs
		utils.EzPickle.__init__(**locals())
		self.reward_function = None
		self.delta_t = 0.2
		self.frame_skip = 2
		self.lines = None
		self.thick = args['wallthickness']

		# initial position + orientation
		self.init_qpos = np.tile(np.array([0.5, 0.5, 0.0]), (self.num_envs, 1))

		# TODO required ? 
		self.reset_states = None
		self.reset_steps = None 

		self.steps = None
		self.done = None
		self.init_qvel = None  # velocities are not used
		self.state = self.init_qpos
		self._obs_dim = 3
		self._action_dim = 1
		high = np.ones(self._action_dim)
		low = -high

		self.single_action_space = spaces.Box(low=low, high=high, dtype=np.float64)
		self.action_space = gym.vector.utils.batch_space(
			self.single_action_space,
			self.num_envs)
		self.max_episode_steps = 30

	def set_init_qpos(self, qpos):
		self.init_qpos = np.tile(np.array(qpos), (self.num_envs, 1))

	def set_reset_states(
		self, reset_states: list, reset_steps: Union[list, None] = None
	):
		self.reset_states = np.array(reset_states)
		if reset_steps is None:
			self.reset_steps = np.zeros((len(reset_states),), dtype=int)
		else:
			self.reset_steps = np.array(reset_steps, dtype=int)

	def reset_done(self, done, *, options=None, seed: Optional[int] = None, infos=None):
		pass

	def reset_model(self):
		# reset state to initial value
		if self.reset_states is None:
			self.state = self.init_qpos.copy()
			self.steps = np.zeros(self.num_envs, dtype=int)
			return {}
		else:
			indices = np.random.choice(len(self.reset_states), self.num_envs)
			self.state = self.reset_states[indices]
			self.steps = self.reset_steps[indices]
			return {"reset_states": indices}

	def common_reset(self):
		return self.reset_model()  # reset state to initial value

	def common_reset_done(self, done):
		# done = self.done
		if not isinstance(done, np.ndarray):
			done = np.asarray(done)
		if self.reset_states is None:
			self.state = np.where(done == 1, self.init_qpos, self.state)
			zeros = np.zeros(self.num_envs, dtype=int)
			self.steps = np.where(done.flatten() == 1, zeros, self.steps)
			return {}
		else:
			indices = np.random.choice(len(self.reset_states), self.num_envs)
			r_state = self.reset_states[indices]
			r_steps = self.reset_steps[indices]
			self.state = np.where(done == 1, r_state, self.state)
			self.steps = np.where(done.flatten() == 1, r_steps, self.steps)
			return {"reset_states": indices}

	def update_state(self, state, action, delta_t):

		MAX_SPEED = 0.5
		MAX_STEER = np.pi
		MIN_STEER = -np.pi

		steer = action[:,0]

		## update x, y, theta
		state[:,0] = state[:,0] + MAX_SPEED*np.cos(state[:,2]) * delta_t
		state[:,1] = state[:,1] + MAX_SPEED*np.sin(state[:,2]) * delta_t
		new_orientation = state[:,2] + steer * delta_t


		## check limit angles
		b_angle_admissible = np.logical_and(new_orientation <= 2.0*np.pi, new_orientation >= -2.0*np.pi).reshape(state[:,2].shape)
		# b_angle_non_admissible = (1. - b_angle_admissible).reshape(state[:,2].shape)

		# state[:, 2] = (new_orientation + np.pi) % (2.0 * np.pi) - np.pi

		state[:,2] = np.where(b_angle_admissible==1, new_orientation, state[:,2])

		return state

	def set_reward_function(self, reward_function):
		self.reward_function = (
			reward_function  # the reward function is not defined by the environment
		)

	def set_frame_skip(self, frame_skip: int = 2):
		self.frame_skip = (
			frame_skip  # a call to step() repeats the action frame_skip times
		)

	def valid_step(self,state,new_state):

		if self.lines is None:
			self.lines = [] #todo optim
			def add_hwall(lines,i,j,t=0):
				lines.append([(i-t,j),(i-t,j+1)])
				if t>0:
					lines.append([(i-t,j+1),(i+t,j+1)])
					lines.append([(i+t,j),(i+t,j+1)])
					lines.append([(i+t,j),(i-t,j)])
			def add_vwall(lines,i,j,t=0):
				lines.append([(i,j-t),(i+1,j-t)])
				if t>0:
					lines.append([(i+1,j-t),(i+1,j+t)])
					lines.append([(i,j+t),(i+1,j+t)])
					lines.append([(i,j+t),(i,j-t)])
			t = self.thick
			for i in range(len(self.grid)):
				for j in range(len(self.grid[i])):
					if self.grid[i][j].walls["top"]:
						add_hwall(self.lines,i,j,t)
					if self.grid[i][j].walls["bottom"]:
						add_hwall(self.lines,i+1,j,t)
					if self.grid[i][j].walls["left"]:
						add_vwall(self.lines,i,j,t)
					if self.grid[i][j].walls["right"]:
						add_vwall(self.lines,i,j+1,t)
		# print("lines = ", self.lines)

		intersection = np.full((self.num_envs,), False)
		for (w1, w2) in self.lines:
			intersection = np.logical_or(
				intersection, intersect(state, new_state, w1, w2)
			)
		intersection = np.expand_dims(intersection, axis=-1)
		# print("intersection = ", intersection)

		return intersection

	def plot(self,ax):
		self.draw(ax, self.thick)
		ax.set_xlim((-self.thick, self.maze_size+self.thick))
		ax.set_ylim((-self.thick, self.maze_size+self.thick))


def default_reward_fun(action, new_obs):
	return np.zeros(new_obs.shape[0],1)


class GMazeDubins(GMazeCommon, gym.Env):
	def __init__(self, num_envs: int = 1):
		super().__init__(num_envs)

		self.set_reward_function(default_reward_fun)

		high = np.ones(self._obs_dim)
		low = -high
		self.single_observation_space = spaces.Box(low, high, dtype=np.float64)
		self.observation_space = gym.vector.utils.batch_space(
			self.single_observation_space,
			self.num_envs)


	def step(self,action: np.ndarray):

		new_state = self.state.copy() 
		for _ in range(self.frame_skip):
			new_state = self.update_state(new_state, action, self.delta_t)
			intersection = self.valid_step(self.state, new_state)
			self.state = self.state * intersection + new_state * np.logical_not(
				intersection
			)

		observation = self.state
		reward = self.reward_function(action, observation).reshape(
			(self.num_envs, 1))
		
		self.steps += 1


		terminated = np.zeros((self.num_envs, 1), dtype=bool)
		truncated = np.asarray(
			(self.steps == self.max_episode_steps), dtype=bool
		).reshape((self.num_envs, 1))

		# TODO chose between both options 
		self.done = np.zeros((observation.shape))
		self.done = np.logical_or(truncated, terminated)

		info = {}
		return (
			observation,
			reward,
			terminated,
			truncated,
			info,
		)

	def state_vector(self):
		return self.state

	def reset_model(self):
		# reset state to initial value
		self.state = self.init_qpos

	def reset(self, *, options=None, seed: Optional[int] = None):
		self.reset_model()
		self.steps = np.zeros(self.num_envs)
		return self.state

	def reset_done(self, done, *, options=None, seed: Optional[int] = None):
		self.state = np.where(self.done == 1, self.init_qpos, self.state)
		zeros = np.zeros(self.num_envs)
		self.steps = np.where(self.done.flatten() == 1, zeros, self.steps)
		return self.state.detach().cpu().numpy()

	def set_state(self, qpos, qvel = None):
		self.state = qpos


def goal_distance(goal_a, goal_b):
	assert goal_a.shape[1] == 2
	assert goal_b.shape[1] == 2
	#print("goal_a.shape = ", goal_a.shape)
	#print("goal_b.shape = ", goal_b.shape)

	return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)


def default_compute_reward(
		achieved_goal: np.ndarray,
		desired_goal: np.ndarray):
	
	distance_threshold = 0.1
	reward_type = "sparse"
	d = goal_distance(achieved_goal, desired_goal)
	if reward_type == "sparse":
		return 1.0 * (d <= distance_threshold)
	else:
		return -d

def default_success_function(achieved_goal, desired_goal):
	distance_threshold = 0.1
	d = goal_distance(achieved_goal, desired_goal)
	return 1.0 * (d <= distance_threshold)


class GToyMazeGoalDubins(GMazeCommon, GoalEnv, utils.EzPickle, ABC):
	def __init__(self, num_envs: int = 1):
		super().__init__(num_envs)

		high = np.ones(self._obs_dim)
		low = -high
		self._achieved_goal_dim = 2
		self._desired_goal_dim = 2
		high_achieved_goal = np.ones(self._achieved_goal_dim)
		low_achieved_goal = -high_achieved_goal
		high_desired_goal = np.ones(self._desired_goal_dim)
		low_desired_goal = -high_desired_goal
		self.single_observation_space = spaces.Dict(
			dict(
				observation=spaces.Box(low, high, dtype=np.float64),
				achieved_goal=spaces.Box(
					low_achieved_goal, high_achieved_goal, dtype=np.float64
				),
				desired_goal=spaces.Box(
					low_desired_goal, high_desired_goal, dtype=np.float64
				),
			)
		)
		self.observation_space = gym.vector.utils.batch_space(
			self.single_observation_space,
			self.num_envs)
		
		self.goal = None

		self.compute_reward = None
		self.set_reward_function(default_compute_reward)

		self._is_success = None

		self.max_episode_steps = np.ones((self.num_envs,1)) * self.max_episode_steps
		# print("self.max_episode_steps.shape = ", self.max_episode_steps.shape)
		# self.set_success_function(default_success_function)

	def project_to_goal_space(self, state):
		return state[:, :2]

	def get_obs_dim(self):
		return self._obs_dim
	def get_full_state_dim(self):
		return self._obs_dim
	def get_goal_dim(self):
		return self._achieved_goal_dim

	def goal_distance(self, goal_a, goal_b):
		# assert goal_a.shape == goal_b.shape
		return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)

	def set_reward_function(self, reward_function):
		self.compute_reward = (  # the name is compute_reward in GoalEnv environments
			reward_function
		)


	def reset(self, *, options=None, seed: Optional[int] = None):
		info = self.common_reset()
		self.goal = self._sample_goal()  # sample goal
		self.steps = np.zeros((self.num_envs,1))
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}, info

	def reset_done(self, done, *, options=None, seed: Optional[int] = None):
		zeros = np.zeros((self.num_envs,1))
		self.steps = np.where(self.done == 1, zeros, self.steps)
		newgoal = self._sample_goal()
		self.goal = np.where(self.done == 1, newgoal, self.goal)
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}, {}

	def set_state(self, state, set_state):

		t_state = state.reshape(self.num_envs, self._obs_dim).copy()
		t_set_state = set_state.reshape(self.num_envs, 1)
		self.state = np.where(t_set_state == 1, t_state, self.state)

	def get_state(self):
		return self.state.copy()

	def get_observation(self):
		return {
			'observation': self.state.copy(),
			'achieved_goal': self.project_to_goal_space(self.state),
			'desired_goal': self.goal.copy(),
		}

	def set_goal(self, goal, set_goal):
		t_goal = goal.reshape(self.num_envs, self._achieved_goal_dim)
		t_set_goal = set_goal.reshape(self.num_envs, 1)
		self.goal = np.where(t_set_goal == 1, t_goal, self.goal)

	def get_goal(self):
		return self.goal

	def set_max_episode_steps(self, max_episode_steps, set_steps):
		if type(max_episode_steps) == np.ndarray:
			t_max_episode_steps = max_episode_steps.reshape(self.num_envs, 1)
			t_set_steps = set_steps
			self.max_episode_steps = np.where(t_set_steps == 1, t_max_episode_steps, self.max_episode_steps)
			new_steps = np.zeros((self.num_envs,1))

			assert self.max_episode_steps.shape[0] == new_steps.shape[0]
			assert self.max_episode_steps.shape[1] == new_steps.shape[1]

			self.steps = np.where(t_set_steps == 1, new_steps, self.steps)
		else:
			t_max_episode_steps = max_episode_steps.reshape(self.num_envs, 1)
			self.max_episode_steps = np.where(set_steps == 1, t_max_episode_steps, self.max_episode_steps)


	def _sample_goal(self):
		return self.project_to_goal_space(np.random.rand(self.num_envs, 2) * 2.0 )

	def step(self,action: np.ndarray):

		new_state = self.state.copy()
		for i in range(self.frame_skip):
			new_state = self.update_state(new_state, action, self.delta_t)
			intersection = self.valid_step(self.state, new_state)
			self.state = np.where(intersection==0, new_state, self.state)

		reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal).reshape(
			(self.num_envs, 1))
		
		self.steps += 1

		truncation = (self.steps >= self.max_episode_steps).reshape(
			(self.num_envs, 1))

		is_success = reward.copy()

		done = np.zeros(is_success.shape)
		terminated = done.copy()

		# truncation = truncation * (1 - is_success)
		info = {'is_success': is_success,
				'done_from_env': done,
				'truncation': truncation}
		self.done = np.maximum(truncation, is_success)
		return (
			{
				'observation': self.state.copy(),
				'achieved_goal': self.project_to_goal_space(self.state),
				'desired_goal': self.goal.copy(),
			},
			reward,
			terminated, 
			truncation, 
			info,
		)