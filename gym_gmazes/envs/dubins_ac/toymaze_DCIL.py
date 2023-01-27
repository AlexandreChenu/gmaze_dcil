# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
from abc import abstractmethod
from typing import Optional
import gym
from typing import Union
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc
from IPython import embed

from .maze.maze import Maze

# from maze.maze import Maze
# from skill_manager_mazeenv import SkillsManager


import matplotlib.pyplot as plt


class GoalEnv(gym.Env):
	"""The GoalEnv class that was migrated from gym (v0.22) to gym-robotics"""

	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		super().reset(seed=seed)
		# Enforce that each GoalEnv uses a Goal-compatible observation space.
		if not isinstance(self.observation_space, gym.spaces.Dict):
			raise error.Error(
				"GoalEnv requires an observation space of type gym.spaces.Dict"
			)
		for key in ["observation", "achieved_goal", "desired_goal"]:
			if key not in self.observation_space.spaces:
				raise error.Error('GoalEnv requires the "{}" key.'.format(key))

	@abstractmethod
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


@torch.no_grad()
def intersect(a, b, c, d):
	x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
	y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
	denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

	criterion1 = denom != 0
	t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
	criterion2 = torch.logical_and(t > 0, t < 1)
	t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
	criterion3 = torch.logical_and(t > 0, t < 1)

	return torch.logical_and(torch.logical_and(criterion1, criterion2), criterion3)


class GMazeCommon(Maze):
	def __init__(self, device: str, num_envs: int = 1):

		args={}
		args['mazesize'] = 3
		args['random_seed'] = 0
		args['mazestandard'] = False
		args['wallthickness'] = 0.1
		args['wallskill'] = True
		args['targetkills'] = True

		super(GMazeCommon,self).__init__(args['mazesize'],args['mazesize'],seed=args['random_seed'],standard=args['mazestandard'])
		self.maze_size = int(args['mazesize'])

		# self.grid[0][1].remove_walls(0,2)
		# self.grid[1][1].remove_walls(0,2)

		# self.empty_grid()
		# # for i in range(self.num_rows):
		# # 	if i != 0:
		# self.grid[0][0].add_walls(0,1)
		# self.grid[1][0].add_walls(1,1)
		#
		# self.grid[0][1].add_walls(0,2)
		# # self.grid[1][1].add_walls(1,2)
		# self.grid[2][1].add_walls(2,2)
		# self.grid[2][1].add_walls(3,2)
				# self.grid[i][1].add_walls(i,0)

		self.num_envs = num_envs
		self.device = device
		utils.EzPickle.__init__(**locals())
		self.reward_function = None
		self.delta_t = 0.2
		self.frame_skip = 2
		self.lines = None

		self.thick = args['wallthickness']

		# initial position + orientation
		self.init_qpos = torch.tensor(
			np.tile(np.array([0.5, 0.5]), (self.num_envs, 1))
		).to(self.device)
		self.steps = None
		self.done = None
		self.init_qvel = None  # velocities are not used
		self.state = self.init_qpos
		self._obs_dim = 2
		self._action_dim = 2
		high = np.ones(self._action_dim)
		low = -high
		self.single_action_space = spaces.Box(low=low, high=high, dtype=np.float64)
		self.action_space = gym.vector.utils.batch_space(
			self.single_action_space,
			self.num_envs)
		self.max_episode_steps = 50

	@abstractmethod
	def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
		pass

	@torch.no_grad()
	def update_state(self, state, action, delta_t):

		action = action/5

		## clip displacements
		delta_x = torch.where(action[:,0]<= 0.1, action[:,0], 0.1)
		delta_y = torch.where(action[:,1]<= 0.1, action[:,1], 0.1)

		## update x, y, theta
		state[:,0] = state[:,0] + delta_x
		state[:,1] = state[:,1] + delta_y

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
		## check if new state is in bound
		# b_new_state_in_bound = self.state_isInBounds(new_state)
		# b_new_state_in_bound = torch.unsqueeze(b_new_state_in_bound, dim=-1).int()
		# print(b_new_state_in_bound)

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

		intersection = torch.full((self.num_envs,), False).to(self.device)
		for (w1, w2) in self.lines:
			intersection = torch.logical_or(
				intersection, intersect(state, new_state, w1, w2)
			)
		intersection = torch.unsqueeze(intersection, dim=-1)
		# print("intersection = ", intersection)

		return intersection

	def plot(self,ax):
		self.draw(ax, self.thick)
		ax.set_xlim((-self.thick, self.maze_size+self.thick))
		ax.set_ylim((-self.thick, self.maze_size+self.thick))


@torch.no_grad()
def default_reward_fun(action, new_obs):
	return torch.zeros(new_obs.shape[0],1)

# @torch.no_grad()
# def goal_distance_(goal_a, goal_b):
# 	assert goal_a.shape[0] == 1
# 	assert goal_b.shape[0] == 1
# 	#print("goal_a.shape = ", goal_a.shape)
# 	#print("goal_b.shape = ", goal_b.shape)
# 	if torch.is_tensor(goal_a):
# 		return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
# 	else:
# 		return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)
#
#
# @torch.no_grad()
# def default_reward_fun(
# 		action, new_obs
# ):
# 	desired_goal = np.array([[1.5, 1.5]])
# 	distance_threshold = 0.1
# 	reward_type = "sparse"
# 	d = goal_distance_(new_obs, desired_goal)
# 	return -d


class GToyMaze(GMazeCommon, gym.Env, utils.EzPickle, ABC):
	def __init__(self, device: str = 'cpu', num_envs: int = 1):
		super().__init__(device, num_envs)

		self.set_reward_function(default_reward_fun)

		high = np.ones(self._obs_dim)
		low = -high
		self.single_observation_space = spaces.Box(low, high, dtype=np.float64)
		self.observation_space = gym.vector.utils.batch_space(
			self.single_observation_space,
			self.num_envs)


	@torch.no_grad()
	def step(self,action: np.ndarray):
		action = action.astype(np.float64)
		action = torch.tensor(action).to(self.device)

		new_state = torch.clone(self.state)
		for i in range(self.frame_skip):
			new_state = self.update_state(new_state, action, self.delta_t)
			intersection = self.valid_step(self.state, new_state)
			self.state = self.state * intersection + new_state * torch.logical_not(
				intersection
			)

		observation = self.state
		reward = self.reward_function(action, observation).reshape(
			(self.num_envs, 1))
		self.steps += 1

		truncation = (self.steps >= self.max_episode_steps).double().reshape(
			(self.num_envs, 1))

		is_success = torch.clone(reward)

		done = torch.zeros(is_success.shape)

		# truncation = truncation * (1 - is_success)
		info = {'is_success': is_success.detach().cpu().numpy(),
				'done_from_env': done.detach().cpu().numpy(),
				'truncation': truncation.detach().cpu().numpy()}
		self.done = torch.maximum(truncation, is_success)

		return (
			observation.detach().cpu().numpy(),
			reward.detach().cpu().numpy(),
			self.done.detach().cpu().numpy(),
			info
		)

	def state_vector(self):
		return self.state

	@torch.no_grad()
	def reset_model(self):
		# reset state to initial value
		self.state = self.init_qpos

	@torch.no_grad()
	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		self.reset_model()
		self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
		return self.state.detach().cpu().numpy()

	@torch.no_grad()
	def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
		self.state = torch.where(self.done == 1, self.init_qpos, self.state)
		zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
		self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)
		return self.state.detach().cpu().numpy()

	@torch.no_grad()
	def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor = None):
		self.state = qpos


@torch.no_grad()
def goal_distance(goal_a, goal_b):
	assert goal_a.shape[1] == 1
	assert goal_b.shape[1] == 1
	#print("goal_a.shape = ", goal_a.shape)
	#print("goal_b.shape = ", goal_b.shape)
	if torch.is_tensor(goal_a):
		return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
	else:
		return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)


@torch.no_grad()
def default_compute_reward(
		achieved_goal: Union[np.ndarray, torch.Tensor],
		desired_goal: Union[np.ndarray, torch.Tensor],
		info: dict
):
	distance_threshold = 0.1
	reward_type = "sparse"
	d = goal_distance(achieved_goal, desired_goal)
	if reward_type == "sparse":
		# if torch.is_tensor(achieved_goal):
		#     return (d < distance_threshold).double()
		# else:
		return 1.0 * (d <= distance_threshold)
		# return -1.0 * (d > distance_threshold)
	else:
		return -d


@torch.no_grad()
def default_success_function(achieved_goal: torch.Tensor, desired_goal: torch.Tensor):
	distance_threshold = 0.1
	d = goal_distance(achieved_goal, desired_goal)
	return 1.0 * (d <= distance_threshold)


class GToyMazeGoal(GMazeCommon, GoalEnv, utils.EzPickle, ABC):
	def __init__(self, device: str = 'cpu', num_envs: int = 1):
		super().__init__(device, num_envs)

		high = np.ones(self._obs_dim)
		low = -high
		self._achieved_goal_dim = 1
		self._desired_goal_dim = 1
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

		self.max_episode_steps = torch.ones((self.num_envs,1)).double()*30.
		# print("self.max_episode_steps.shape = ", self.max_episode_steps.shape)
		# self.set_success_function(default_success_function)

	@torch.no_grad()
	def project_to_goal_space(self, state):
		return state[:, 1:]

	def get_obs_dim(self):
		return self._obs_dim
	def get_full_state_dim(self):
		return self._obs_dim
	def get_goal_dim(self):
		return self._achieved_goal_dim

	@torch.no_grad()
	def goal_distance(self, goal_a, goal_b):
		# assert goal_a.shape == goal_b.shape
		if torch.is_tensor(goal_a):
			return torch.linalg.norm(goal_a[:,:] - goal_b[:, :], axis=-1)
		else:
			return np.linalg.norm(goal_a[:, :] - goal_b[:, :], axis=-1)

	@torch.no_grad()
	def set_reward_function(self, reward_function):
		self.compute_reward = (  # the name is compute_reward in GoalEnv environments
			reward_function
		)

	@torch.no_grad()
	def reset_model(self):
		# reset state to initial value
		self.state = torch.clone(self.init_qpos)

	@torch.no_grad()
	def set_state(self, state, set_state):
		# print(type(state) == np.ndarray)
		# print("\nstate.shape = ", state.shape)
		# print("state = ", state)
		# print("self.state.shape = ", self.state.shape)
		# print("self.state = ", self.state)
		# print("set_state.shape = ", set_state)
		# print("set_state = ", set_state)
		if type(state) == np.ndarray:
			t_state = torch.from_numpy(state.reshape(self.num_envs, self._obs_dim).copy())
			t_set_state = torch.from_numpy(set_state).reshape(self.num_envs, 1)
			self.state = torch.where(t_set_state == 1, t_state, self.state)
		else:
			t_state = torch.clone(state.reshape(self.num_envs, self._obs_dim))
			self.state = torch.where(set_state == 1, t_state, self.state)

	@torch.no_grad()
	def get_state(self):
		return self.state.detach().cpu().numpy()

	@torch.no_grad()
	def get_observation(self):
		return {
			'observation': self.state.detach().cpu().numpy(),
			'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
			'desired_goal': self.goal.detach().cpu().numpy(),
		}

	@torch.no_grad()
	def set_goal(self, goal, set_goal):
		if type(goal) == np.ndarray:
			t_goal = torch.from_numpy(goal.reshape(self.num_envs, self._achieved_goal_dim).copy())
			t_set_goal = torch.from_numpy(set_goal).reshape(self.num_envs, 1)
			self.goal = torch.where(t_set_goal == 1, t_goal, self.goal)
		else:
			t_goal = torch.clone(goal.reshape(self.num_envs, self._achieved_goal_dim))
			self.goal = torch.where(set_goal == 1, t_goal, self.goal)

	@torch.no_grad()
	def get_goal(self):
		return self.goal.detach().cpu().numpy()

	@torch.no_grad()
	def set_max_episode_steps(self, max_episode_steps, set_steps):
		if type(max_episode_steps) == np.ndarray:
			t_max_episode_steps = torch.from_numpy(max_episode_steps.reshape(self.num_envs, 1).copy())
			# print("t_max_episode_steps.shape = ", t_max_episode_steps.shape)
			t_set_steps = torch.from_numpy(set_steps)
			# print("t_set_steps.shape = ", t_set_steps.shape)
			self.max_episode_steps = torch.where(t_set_steps == 1, t_max_episode_steps, self.max_episode_steps)
			new_steps = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)

			# print("self.max_episode_steps.shape = ", self.max_episode_steps.shape)
			# print("new_steps.shape = ", new_steps.shape)

			assert self.max_episode_steps.shape[0] == new_steps.shape[0]
			assert self.max_episode_steps.shape[1] == new_steps.shape[1]

			self.steps = torch.where(t_set_steps == 1, new_steps, self.steps)
		else:
			t_max_episode_steps = torch.clone(max_episode_steps.reshape(self.num_envs, 1))
			self.max_episode_steps = torch.where(set_steps == 1, t_max_episode_steps, self.max_episode_steps)


	@torch.no_grad()
	def _sample_goal(self):
		return (torch.rand(self.num_envs, self._desired_goal_dim) * 2.0 ).reshape(self.num_envs, self._achieved_goal_dim).double().to(self.device)


	@torch.no_grad()
	def reset(self, options=None, seed: Optional[int] = None, infos=None):
		self.reset_model()  # reset state to initial value
		self.goal = self._sample_goal()  # sample goal
		self.steps = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
		return {
			'observation': self.state.detach().cpu().numpy(),
			'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
			'desired_goal': self.goal.detach().cpu().numpy(),
		}

	@torch.no_grad()
	def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
		# self.state = torch.where(self.done == 1, self.init_qpos, self.state)
		zeros = torch.zeros((self.num_envs,1), dtype=torch.int).to(self.device)
		self.steps = torch.where(self.done == 1, zeros, self.steps)
		newgoal = self._sample_goal()
		self.goal = torch.where(self.done == 1, newgoal, self.goal)
		return {
			'observation': self.state.detach().cpu().numpy(),
			'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
			'desired_goal': self.goal.detach().cpu().numpy(),
		}

	@torch.no_grad()
	def step(self,action: np.ndarray):
		action = action.astype(np.float64)

		action = torch.tensor(action).to(self.device)

		new_state = torch.clone(self.state)
		for i in range(self.frame_skip):
			new_state = self.update_state(new_state, action, self.delta_t)

			intersection = self.valid_step(self.state, new_state)

			# self.state = self.state * intersection + new_state * torch.logical_not(
			# 	intersection
			# )

			self.state = torch.where(intersection==0, new_state, self.state)

		reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal, {}).reshape(
			(self.num_envs, 1))

		self.steps += 1

		truncation = (self.steps >= self.max_episode_steps).double().reshape(
			(self.num_envs, 1))

		is_success = torch.clone(reward)

		# done = torch.zeros(is_success.shape)
		done = torch.tensor(intersection)

		# truncation = truncation * (1 - is_success)
		info = {'is_success': is_success.detach().cpu().numpy(),
				'done_from_env': done.detach().cpu().numpy(),
				'truncation': truncation.detach().cpu().numpy()}
		self.done = torch.logical_or(done, torch.maximum(truncation, is_success))
		return (
			{
				'observation': self.state.detach().cpu().numpy(),
				'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
				'desired_goal': self.goal.detach().cpu().numpy(),
			},
			reward.detach().cpu().numpy(),
			self.done.detach().cpu().numpy(),
			info,
		)


if __name__ == "__main__":

	env = GToyMazeGoal()
	obs = env.reset()
	print(env.goal)
	trajs = []
	traj = [obs]
	n_steps = 500

	print(env.grid)

	# for i in range(n_steps):
	# 	traj.append(obs)
	# 	a = env.action_space.sample()
	# 	obs,_,d,_ = env.step(a)
	#
	# 	if d.max():
	# 		env.reset()
	# 		trajs.append(traj)
	# 		traj = []

	fig, ax = plt.subplots()
	env.plot(ax)
	#
	# for traj in trajs:
	# 	X = [obs["observation"][0,0] for obs in traj]
	# 	Y = [obs["observation"][0,1] for obs in traj]
	# 	ax.plot(X,Y)

	plt.show()
