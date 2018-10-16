from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from gym_super_mario_bros import SuperMarioBrosEnv
import gym_super_mario_bros

Moves = [
    ['NOP'],
    ['right'],
    ['right', 'A'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['down'],
    ['up'],
]

x_factor = 5;
cost_of_death = -25;
single_reward_range = (-25, 25)

@property
def x_reward(self):
        _reward = self._x_position - self._x_position_last
        self._x_position_last = self._x_position
        return _reward * self._x_factor

def reward_all(self):
        return self._x_reward + self._time_penalty + self._death_penalty

@property
def death_cost(self):
	if self._is_dying or self._is_dead:
		return self.cost_of_death
	return 0

SuperMarioBrosEnv._x_factor = x_factor
SuperMarioBrosEnv._cost_of_death = cost_of_death
SuperMarioBrosEnv.reward_range = single_reward_range
SuperMarioBrosEnv._x_reward = x_reward
SuperMarioBrosEnv._death_penalty = death_cost
SuperMarioBrosEnv._get_reward = reward_all

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v1")	#Same as gym.make
env = BinarySpaceToDiscreteSpaceEnv(env, Moves)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    env.render()

env.close()