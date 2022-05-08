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
from .skill_manager_mazeenv import SkillsManager


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
        args['mazesize'] = 5
        args['random_seed'] = 0
        args['mazestandard'] = False
        args['wallthickness'] = 0.1
        args['wallskill'] = True
        args['targetkills'] = True

        super(GMazeCommon,self).__init__(args['mazesize'],args['mazesize'],seed=args['random_seed'],standard=args['mazestandard'])
        self.maze_size = int(args['mazesize'])

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
            np.tile(np.array([0.5, 0.5, 0.0]), (self.num_envs, 1))
        ).to(self.device)
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
        self.max_episode_steps = 50

    @abstractmethod
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
        pass

    @torch.no_grad()
    def update_state(self, state, action, delta_t):

        MAX_SPEED = 0.5
        MAX_STEER = torch.pi
        MIN_STEER = -torch.pi

        steer = action[:,0]

        ## update x, y, theta
        state[:,0] = state[:,0] + MAX_SPEED*torch.cos(state[:,2]) * delta_t
        state[:,1] = state[:,1] + MAX_SPEED*torch.sin(state[:,2]) * delta_t
        new_orientation = state[:,2] + steer * delta_t

        ## check limit angles
        # b_angle_admissible = torch.logical_and(new_orientation <= MAX_STEER, new_orientation >= MIN_STEER).reshape(state[:,2].shape).double()
        # b_angle_non_admissible = (1. - b_angle_admissible).reshape(state[:,2].shape)

        state[:, 2] = (new_orientation + np.pi) % (2.0 * np.pi) - np.pi

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


class GMazeDubins(GMazeCommon, gym.Env, utils.EzPickle, ABC):
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
        self.done = torch.zeros((observation.shape))
        info = {}
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
    assert goal_a.shape[1] == 2
    assert goal_b.shape[1] == 2
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
        return 1.0 * (d < distance_threshold)
    else:
        return -d


@torch.no_grad()
def default_success_function(achieved_goal: torch.Tensor, desired_goal: torch.Tensor):
    distance_threshold = 0.1
    d = goal_distance(achieved_goal, desired_goal)
    return 1.0 * (d < distance_threshold)


class GMazeGoalDubins(GMazeCommon, GoalEnv, utils.EzPickle, ABC):
    def __init__(self, device: str = 'cpu', num_envs: int = 1):
        super().__init__(device, num_envs)

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
        # self.set_success_function(default_success_function)

    @torch.no_grad()
    def project_to_goal_space(self, state):
        return state[:, :2]

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

    ### TODO: replace with skill manager
    @torch.no_grad()
    def _sample_goal(self):
        # return (torch.rand(self.num_envs, 2) * 2. - 1).to(self.device)
        return self.project_to_goal_space(torch.rand(self.num_envs, 2) * 2.0 ).to(self.device)
        # return torch.tensor(
        #     np.tile(np.array([1.78794995, 1.23542976]), (self.num_envs, 1))
        # ).to(self.device)

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):
        self.reset_model()  # reset state to initial value
        self.goal = self._sample_goal()  # sample goal
        self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
        self.state = torch.where(self.done == 1, self.init_qpos, self.state)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)
        newgoal = self._sample_goal()
        self.goal = torch.where(self.done == 1, newgoal, self.goal)
        return {
            'observation': self.state.detach().cpu().numpy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
            'desired_goal': self.goal.detach().cpu().numpy(),
        }

    @torch.no_grad()
    def step(self,action: np.ndarray):
        action = torch.tensor(action).to(self.device)

        new_state = torch.clone(self.state)
        for i in range(self.frame_skip):
            new_state = self.update_state(new_state, action, self.delta_t)

        intersection = self.valid_step(self.state, new_state)

        self.state = self.state * intersection + new_state * torch.logical_not(
            intersection
        )

        reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal, {}).reshape(
            (self.num_envs, 1))
        self.steps += 1
        truncation = (self.steps >= self.max_episode_steps).double().reshape(
            (self.num_envs, 1))

        is_success = torch.clone(reward)

        truncation = truncation * (1 - is_success)
        info = {'is_success': is_success.detach().cpu().numpy(),
                'truncation': truncation.detach().cpu().numpy()}
        self.done = torch.maximum(truncation, is_success)
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


class GMazeDCILDubins(GMazeGoalDubins):
    def __init__(self, demo_path, device: str = 'cpu', num_envs: int = 1):
        super().__init__(device, num_envs)

        self.done = torch.ones((self.num_envs, 1)).int().to(self.device)

        ## fake init as each variable is modified after first reset
        self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.is_success = torch.zeros((self.num_envs, 1)).int().to(self.device)
        self.goal = self.project_to_goal_space(torch.clone(self.state)).to(self.device)

        self.truncation = None

        self.do_overshoot = True

        self.demo_path = demo_path
        self.skill_manager = SkillsManager(self.demo_path, self) ## skill length in time-steps
        # self.skill_manager = SkillsManager("/Users/chenu/Desktop/PhD/github/dcil/demos/toy_dubinsmazeenv/1.demo", self)

    @torch.no_grad()
    def step(self,action: np.ndarray):
        action = torch.tensor(action).to(self.device)

        new_state = torch.clone(self.state)
        for i in range(self.frame_skip):
            new_state = self.update_state(new_state, action, self.delta_t)

        intersection = self.valid_step(self.state, new_state)

        self.state = self.state * intersection + new_state * torch.logical_not(
            intersection
        )

        reward = self.compute_reward(self.project_to_goal_space(self.state), self.goal, {}).reshape(
            (self.num_envs, 1))
        self.steps += 1

        # print("self.steps = ", self.steps)
        # print("self.max_episode_steps = ", self.max_episode_steps)
        truncation = (self.steps >= self.max_episode_steps.view(self.steps.shape)).double().reshape(
            (self.num_envs, 1))

        #truncation = torch.zeros((self.num_envs, 1))

        # print("truncation = ", truncation)

        is_success = torch.clone(reward)/1.
        self.is_success = torch.clone(is_success)

        # print("self.is_success = ", self.is_success)

        truncation = truncation * (1 - is_success)
        info = {'is_success': torch.clone(is_success).detach().cpu().numpy(),
                'truncation': torch.clone(truncation).detach().cpu().numpy()}
        self.done = torch.maximum(truncation, is_success)

        ## get next goal and next goal availability boolean
        next_goal_state, info['next_goal_avail'] = self.skill_manager.next_goal()
        info['next_goal'] = self.project_to_goal_space(next_goal_state)

        return (
            {
                'observation': self.state.detach().cpu().numpy().copy(),
                'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy().copy(),
                'desired_goal': self.goal.detach().cpu().numpy().copy(),
            },
            reward.detach().cpu().numpy().copy(),
            self.done.detach().cpu().numpy().copy(),
            info,
        )


    # @torch.no_grad()
    # def reset_model(self):
    #     # reset state to initial value
    #     self.state = self.init_qpos

    # @torch.no_grad()
    # def reset(self, options=None, seed: Optional[int] = None, infos=None):
    #     self.reset_model()  # reset state to initial value
    #     self.goal = self._sample_goal()  # sample goal
    #     self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
    #     return {
    #         'observation': self.state.detach().cpu().numpy(),
    #         'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
    #         'desired_goal': self.goal.detach().cpu().numpy(),
    #     }

    ## TODO:
    ## - check if overshoot available
    ## - adapt reset w/ or w/o overshoot
    ## - goal sampling

    # @torch.no_grad()
    # def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
    #     self.state = torch.where(self.done == 1, self.init_qpos, self.state)
    #     zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
    #     self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)
    #     newgoal = self._sample_goal()
    #     self.goal = torch.where(self.done == 1, newgoal, self.goal)
    #     return {
    #         'observation': self.state.detach().cpu().numpy(),
    #         'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy(),
    #         'desired_goal': self.goal.detach().cpu().numpy(),
    #     }

    def set_skill(self, skill_indx):
        start_state, length_skill, goal_state = self.skill_manager.set_skill(skill_indx)
        goal = self.project_to_goal_space(goal_state)
        self.state = start_state
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = zeros
        self.goal = goal

        return {
            'observation': self.state.detach().cpu().numpy().copy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy().copy(),
            'desired_goal': self.goal.detach().cpu().numpy().copy(),
        }

    def shift_goal(self):
        goal_state, _ = self.skill_manager.shift_goal()
        goal = self.project_to_goal_space(goal_state)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = zeros
        self.goal = goal

        return {
            'observation': self.state.detach().cpu().numpy().copy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy().copy(),
            'desired_goal': self.goal.detach().cpu().numpy().copy(),
        }

    @torch.no_grad()
    def _select_skill(self):
        ## done indicates indx to change
        ## overshoot indicates indx to shift by one
        ## is success indicates if we should overshoot
        return self.skill_manager._select_skill(torch.clone(self.done.int()), torch.clone(self.is_success.int()), do_overshoot = self.do_overshoot)

    @torch.no_grad()
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
        # print("\n reset_done")
        start_state, length_skill, goal_state, b_overshoot_possible = self._select_skill()
        goal = self.project_to_goal_space(goal_state)

        ## update successes and failures
        for indx_env in range(self.num_envs):
            if self.done[indx_env] == 1:
                if self.is_success[indx_env] == 1:
                    self.skill_manager.add_success(self.skill_manager.indx_goal[indx_env])
                else:
                    self.skill_manager.add_failure(self.skill_manager.indx_goal[indx_env])

        b_change_state = torch.logical_and(self.done, torch.logical_not(b_overshoot_possible)).int()

        self.state = torch.where(b_change_state == 1, start_state, self.state)
        # self.state = torch.where(self.done == 1, start_state, self.state)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)

        # self.max_episode_steps = torch.where(self.done == 1, length_skill, self.max_episode_steps)

        self.goal = torch.where(self.done == 1, goal, self.goal).to(self.device)

        return {
            'observation': self.state.detach().cpu().numpy().copy(),
            'achieved_goal': self.project_to_goal_space(self.state).detach().cpu().numpy().copy(),
            'desired_goal': self.goal.detach().cpu().numpy().copy(),
        }

    @torch.no_grad()
    def reset(self, options=None, seed: Optional[int] = None, infos=None):

        skill_indx = torch.ones((self.num_envs,))
        obs = self.set_skill(skill_indx)
        zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        self.max_episode_steps = torch.ones(self.num_envs, dtype=torch.int).to(self.device)*10
        self.steps = zeros

        return obs


if (__name__=='__main__'):

    ## Test GMazeCommon

    # env = GMazeCommon(device="cpu", num_envs=2)
    #
    # state = env.state
    # new_state = torch.tensor([[0.6000, 0.6000, 0.0000],
    #                           [-0.1000, 0.1000, 0.0000]])
    #
    # env.valid_step(state, new_state)

    ## Test GMazeGoalDubins

    # traj = []
    #
    # env = GMazeGoalDubins(device="cpu", num_envs=6)
    # print("state = ", env.state)
    #
    # env.reset()
    #
    # for i in range(70):
    #     traj.append(env.state)
    #     action = env.action_space.sample()
    #     state, reward, done, info = env.step(action)
    #     # print("state = ", state)
    #     print("env.steps = ", env.steps)
    #
    # env.reset()
    # print("env.steps = ", env.steps)
    #
    # fig, ax = plt.subplots()
    # env.plot(ax)
    # for i in range(traj[0].shape[0]):
    #     X = [state[i][0] for state in traj]
    #     Y = [state[i][1] for state in traj]
    #     ax.plot(X,Y)
    #
    # plt.show()

    traj = []

    env = GMazeDCILDubins(device="cpu", num_envs=6)
    # print("state = ", env.state)

    print("states = ", env.skill_manager.states)

    state = env.reset()
    # print("state (after reset) = ", state)

    for i in range(300):
        traj.append(env.state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

        env.reset()
        # print("state = ", state)
        # print("env.steps = ", env.steps)

    env.reset()
    # print("env.steps = ", env.steps)

    fig, ax = plt.subplots()
    env.plot(ax)
    for i in range(traj[0].shape[0]):
        X = [state[i][0] for state in traj]
        Y = [state[i][1] for state in traj]
        Theta = [state[i][2] for state in traj]
        ax.scatter(X,Y, marker=".")

        for x, y, t in zip(X,Y,Theta):
            dx = np.cos(t)
            dy = np.sin(t)
            arrow = plt.arrow(x,y,dx*0.1,dy*0.1,alpha = 0.6,width = 0.01, zorder=6)

    circles = []
    for state in env.skill_manager.L_states:
        circle = plt.Circle((state[0][0], state[0][1]), 0.1, color='m', alpha = 0.6)
        circles.append(circle)
        # ax.add_patch(circle)
    coll = mc.PatchCollection(circles, color="plum", zorder = 4)
    ax.add_collection(coll)

    plt.show()
