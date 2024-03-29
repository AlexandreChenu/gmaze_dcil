import os
import gym
from gym.envs.registration import register

__version__ = "0.1.0"


def envpath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


print("gym-gmazes: ")
print("|    gym version and path:", gym.__version__, gym.__path__)

print("|    REGISTERING GMazeDubins-v0 from", envpath())
register(
    id="GMazeDubins-v0",
    entry_point="gym_gmazes.envs:GMazeDubins",
)

print("|    REGISTERING GMazeGoalDubins-v0 from", envpath())
register(
    id="GMazeGoalDubins-v0",
    entry_point="gym_gmazes.envs:GMazeGoalDubins",
)

print("|    REGISTERING GMazeGoalEmptyDubins-v0 from", envpath())
register(
    id="GMazeGoalEmptyDubins-v0",
    entry_point="gym_gmazes.envs:GMazeGoalEmptyDubins",
)

# print("|    REGISTERING GMazeDCILDubins-v0 from", envpath())
# register(
#     id="GMazeDCILDubins-v0",
#     entry_point="gym_gmazes.envs:GMazeDCILDubins",
# )

print("|    REGISTERING GToyMaze-v0 from", envpath())
register(
    id="GToyMaze-v0",
    entry_point="gym_gmazes.envs:GToyMaze",
)

print("|    REGISTERING GToyMazeGoal-v0 from", envpath())
register(
    id="GToyMazeGoal-v0",
    entry_point="gym_gmazes.envs:GToyMazeGoal",
)

print("|    REGISTERING GToyMazeGoalDubins-v0 from", envpath())
register(
    id="GToyMazeGoalDubins-v0",
    entry_point="gym_gmazes.envs:GToyMazeGoalDubins",
)
