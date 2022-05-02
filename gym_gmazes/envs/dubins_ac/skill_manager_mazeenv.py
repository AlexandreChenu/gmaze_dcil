import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle




class SkillsManager():

	def __init__(self, demo_path, env):

		self.env = env
		self.num_envs = env.num_envs
		self.device = env.device

		self.eps_state = 1. ## threshold distance in goal space for skill construction
		self.beta = 1.25

		self.L_full_demonstration = self.extract_from_demo(demo_path)
		self.L_states, self.L_budgets = self.clean_demo(self.L_full_demonstration)

		# self.L_states = self.L_states[:2]
		# self.L_budgets = self.L_budgets[:1]

		self.nb_skills = len(self.L_states)-1
		self.states = torch.stack(self.L_states) ## tensor versions
		self.budgets = torch.stack(self.L_budgets)

		## init indx for start and goal states
		self.indx_start = torch.zeros((self.num_envs,1)).int().to(self.device)
		self.indx_goal = torch.ones((self.num_envs,1)).int().to(self.device)

		## a list of list of results per skill
		self.L_skills_results = [[] for _ in self.L_states]
		self.L_overshoot_results = [[[]] for _ in self.L_states]

		self.skill_window = 20
		self.max_size_starting_state_set = 100

		self.weighted_sampling = True

		self.delta_step = 1
		self.dist_threshold = 0.1

	def extract_from_demo(self, demo_path, verbose=1):
		"""
		Extract demo from path
		"""
		L_states = []

		if verbose:
			print("filename :\n", demo_path)
		if not os.path.isfile(demo_path):
			print ("File does not exist.")
			return

		with open(demo_path, "rb") as f:
			demo = pickle.load(f)
		for state, action in zip(demo["obs"], demo["actions"]):
			L_states.append(torch.tensor(np.tile(np.array(state)[:3], (self.num_envs, 1))))
		return L_states

	def clean_demo(self, L_states):
		"""
		Clean the demonstration trajectory according to hyperparameter eps_dist.
		"""
		L_states_clean = [L_states[0]]
		L_budgets = []

		i = 0
		while i < len(L_states)-1:
			k = 1
			sum_dist = 0

			# cumulative distance
			while sum_dist <= self.eps_state and i + k < len(L_states) - 1:
				sum_dist += self.env.goal_distance(self.env.project_to_goal_space(L_states[i+k]), self.env.project_to_goal_space(L_states[i+k-1]))[0]
				k += 1
			if sum_dist > self.eps_state or i + k == len(L_states) - 1:
				L_states_clean.append(L_states[i+k])

			# L_budgets.append(int(self.beta*k))
			L_budgets.append(torch.tensor(np.tile(np.array([int(self.beta*k)]), (self.num_envs, 1))))

			i = i + k

		return L_states_clean, L_budgets


	def get_skill(self, skill_indx):
		"""
		Get starting state, length and goal associated to a given skill
		"""
		assert torch.sum((skill_indx < 0).int()) == 0.
		assert torch.sum((skill_indx > len(self.L_states)) == 0.)

		indx_start = skill_indx - self.delta_step
		indx_goal = skill_indx

		length_skill = self.budgets[indx_start.view(-1).long(),0,:]

		starting_state = self.get_starting_state(indx_start)
		goal_state = self.get_goal_state(indx_goal)

		return starting_state, length_skill, goal_state

	def set_skill(self, skill_indx):
		"""
		Get starting state, length and goal associated to a given skill
		"""
		assert torch.sum((skill_indx < 0).int()) == 0.
		assert torch.sum((skill_indx > len(self.L_states)) == 0.)

		self.indx_start = skill_indx - self.delta_step
		self.indx_goal = skill_indx

		length_skill = self.budgets[self.indx_start.view(-1).long(),0,:]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state

	def get_starting_state(self, indx_start):
		return torch.clone(self.states[indx_start.long().view(-1),0,:])

	def get_goal_state(self, indx_goal):
		return torch.clone(self.states[indx_goal.long().view(-1),0,:])

	def add_success_overshoot(self,skill_indx):
		self.L_overshoot_feasible[skill_indx-1]=True
		return

	def add_success(self, skill_indx):
		"""
		Monitor successes for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(1)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def add_failure(self, skill_indx):
		"""
		Monitor failures for a given skill (to bias skill selection)
		"""
		self.L_skills_results[skill_indx].append(0)

		if len(self.L_skills_results[skill_indx]) > self.skill_window:
			self.L_skills_results[skill_indx].pop(0)

		return

	def get_skill_success_rate(self, skill_indx):

		nb_skills_success = self.L_skills_results[skill_indx].count(1)
		s_r = float(nb_skills_success/len(self.L_skills_results[skill_indx]))

		## keep a small probability for successful skills to be selected in order
		## to avoid catastrophic forgetting
		if s_r <= 0.1:
			s_r = 10
		else:
			s_r = 1./s_r

		return s_r

	def get_skills_success_rates(self):

		L_rates = []
		for i in range(self.delta_step, len(self.L_states)):
			L_rates.append(self.get_skill_success_rate(i))

		return L_rates


	def sample_skill_indx(self):
		"""
		Sample a skill indx.

		2 cases:
			- weighted sampling of skill according to skill success rates
			- uniform sampling
		"""
		weights_available = True
		for i in range(self.delta_step,len(self.L_skills_results)):
			if len(self.L_skills_results[i]) == 0:
				weights_available = False

		# print("self.L_skills_results = ", self.L_skills_results)
		#print("weights available = ", weights_available)

		## fitness based selection
		if self.weighted_sampling and weights_available:

			L_rates = self.get_skills_success_rates()

			assert len(L_rates) == len(self.L_states) - self.delta_step

			## weighted sampling
			total_rate = sum(L_rates)


			L_new_skill_indx = []
			for i in range(self.num_envs):
				pick = random.uniform(0, total_rate)

				current = 0
				for i in range(0,len(L_rates)):
					s_r = L_rates[i]
					current += s_r
					if current > pick:
						break

				i += self.delta_step
				L_new_skill_indx.append([i])

			## TODO: switch for tensor version
			new_skill_indx = torch.tensor(L_new_skill_indx)

			assert new_skill_indx.shape == (self.num_envs, 1)

		## uniform sampling
		else:
			new_skill_indx = torch.randint(1, self.nb_skills+1, (self.num_envs, 1))

		return new_skill_indx.int()

	def shift_goal(self):
		"""
		Returns next goal state corresponding
		"""
		cur_indx = torch.clone(self.indx_goal)
		next_indx = cur_indx + 1
		next_skill_avail = (next_indx <= self.nb_skills).int()
		next_skill_indx = torch.where(next_skill_avail == 1, next_indx, cur_indx)
		self.indx_goal = next_skill_indx
		next_goal_state = self.get_goal_state(self.indx_goal)

		return next_goal_state, next_skill_avail

	def next_goal(self):
		"""
		Returns next goal state corresponding
		"""
		cur_indx = torch.clone(self.indx_goal)
		next_indx = cur_indx + 1
		next_skill_avail = (next_indx <= self.nb_skills).int()
		next_skill_indx = torch.where(next_skill_avail == 1, next_indx, cur_indx)
		next_goal_state = self.get_goal_state(next_skill_indx)

		return next_goal_state, next_skill_avail


	def next_skill_indx(self, cur_indx):
		"""
		Shift skill indices by one and assess if skill indices are available
		"""

		next_indx = cur_indx + 1
		next_skill_avail = (next_indx <= self.nb_skills).int()

		return next_indx.int(), next_skill_avail.int()

	def _select_skill(self, done, is_success, init=False, do_overshoot = True):
		"""
		Select skills (starting state, budget and goal)
		"""

		sampled_skill_indx = self.sample_skill_indx() ## return tensor of new indices
		next_skill_indx, next_skill_avail = self.next_skill_indx(torch.clone(self.indx_goal)) ## return tensor of next skills indices

		# print("next_skill_avail = ", next_skill_avail)
		## check if overshoot is possible (success + shifted skill avail)

		# print("overshoot_possible = ", overshoot_possible)
		#
		# print("next_skill_indx = ", next_skill_indx)
		# print("sampled_skill_indx = ", sampled_skill_indx)
		overshoot_possible = torch.logical_and(is_success, next_skill_avail).int() * (1 - (not do_overshoot))

		## if overshoot possible, choose next skill indx, otherwise, sample new skill indx
		new_skill_indx = torch.where(overshoot_possible == 1, next_skill_indx, sampled_skill_indx)

		# print("new_skill_indx = ", new_skill_indx)
		# print("done = ", done)
		# print("before self.indx_goal = ", self.indx_goal)

		self.indx_goal = torch.where(done == 1, new_skill_indx, self.indx_goal.int())

		# print("after self.indx_goal = ", self.indx_goal)

		## skill indx coorespond to a goal indx
		self.indx_start = (self.indx_goal - self.delta_step)
		# print("self.indx_start = ", self.indx_start.view(-1))
		length_skill = self.budgets[self.indx_start.view(-1).long(),0,:]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state, overshoot_possible


	def _random_skill(self):
		"""
		Select skills (starting state, budget and goal)
		"""

		sampled_skill_indx = self.sample_skill_indx() ## return tensor of new indices

		self.indx_goal = sampled_skill_indx

		# print("self.indx_goal = ", self.indx_goal)

		## skill indx coorespond to a goal indx
		self.indx_start = (self.indx_goal - self.delta_step)
		# print("self.indx_start = ", self.indx_start.view(-1))
		length_skill = self.budgets[self.indx_start.view(-1).long(),0,:]

		starting_state = self.get_starting_state(self.indx_start)
		goal_state = self.get_goal_state(self.indx_goal)

		return starting_state, length_skill, goal_state



if (__name__=='__main__'):

	from dubins_DCIL import GMazeGoalDubins
	# env = GMazeCommon(device="cpu", num_envs=2)
	#
	# state = env.state
	# new_state = torch.tensor([[0.6000, 0.6000, 0.0000],
	#                           [-0.1000, 0.1000, 0.0000]])
	#
	# env.valid_step(state, new_state)

	traj = []

	env = GMazeGoalDubins(device="cpu", num_envs=6)
	env.reset()

	demo_path = "/Users/chenu/Desktop/PhD/github/dcil/demos/toy_dubinsmazeenv/1.demo"
	sm = SkillsManager(demo_path, env)

	print("length full demo = ", len(sm.L_full_demonstration))
	print("length cleaned demo = ", len(sm.L_states))
	print("length budgets = ", len(sm.L_budgets))

	# print("states = ", sm.states)
	print("states.shape = ", sm.states.shape)
	# print("budget = ", sm.budgets)
	print("budget.shape = ", sm.budgets.shape)

	sm._random_skill()

	print("start indx = ", sm.indx_start)
	print("goal indx = ", sm.indx_goal)



	next_skill_indx, next_skill_avail = sm.next_skill_indx()

	print("next skill indx = ", next_skill_indx)
	print("next skill avail = ", next_skill_avail)

	sampled_skill_indx = sm.sample_skill_indx()

	print("sampled skill indx = ", sampled_skill_indx)

	print("states = ", sm.states[:,0,:])

	start_state, budget, goal_state = sm.get_skill(sampled_skill_indx)

	print("budget = ", budget)
	print("goal_state = ", goal_state)
	print("start_state = ", start_state)

	# print(env.state)
