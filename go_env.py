"""
Go environment using gym.
"""
import gym
from gym import logger
from gym.utils import seeding
import numpy as np
from go import Board


class GoEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, size):
        self.board = Board(size)
        self.size = size
        self.np_random = None
        self.action_size = 2
        self.state_size = (size, size, 2)
        self.seed()

        self.viewer = None
        self.steps_beyond_done = None

    def __getstate__(self):
        state = np.zeros((self.size, self.size, 2))
        for width in range(self.size):
            for height in range(self.size):
                if self.board._state[0][width][height] == self.board.WHITE:
                    state[width, height, 0] = 1
                elif self.board._state[0][width][height] == self.board.BLACK:
                    state[width, height, 1] = 1
                # elif
        return state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x, y = action
        self.board.move(x, y)
        '''--------------------------------------------------------'''

        done = self.board.isgameend()
        # done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.__getstate__()), reward, done, {}

    def reset(self):
        self.board = Board(self.size)
        return self.__getstate__()

    def render(self, mode='human'):
        pass
