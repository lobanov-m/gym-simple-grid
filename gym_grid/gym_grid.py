import arcturus as ar
import cv2
import gym
import gym.utils.seeding
import numpy as np
from enum import IntEnum
from typing import Optional

class GymGrid(gym.Env):
    class Actions(IntEnum):
        Up = 0
        UpRight = 1
        Right = 2
        DownRight = 3
        Down = 4
        DownLeft = 5
        Left = 6
        UpLeft = 7
        Stay = 8

    def __init__(self, grid_size=(8, 8), max_steps=100, seed=1):
        self.grid_size = (grid_size[0] + 2, grid_size[1] + 2)
        self.max_steps = max_steps

        self.actions = GymGrid.Actions
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.grid = None
        self.position = None
        self.target = None
        self.i_step = None

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size[1], self.grid_size[0], 3),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })

        self.np_random = None
        self.seed(seed=seed)
        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.grid = np.ones((self.grid_size[1], self.grid_size[0]), np.uint8) * 255
        self.grid[0] = 0
        self.grid[-1] = 0
        self.grid[:, 0] = 0
        self.grid[:, -1] = 0

        # self.position = self._random_position()
        self.position = np.array([1, 1], np.int64)
        self.target = np.array([self.grid_size[0] - 2, self.grid_size[1] - 2], np.int64)

        self.i_step = 0

        obs = self._get_observation()
        return obs

    def step(self, action):
        self.i_step += 1
        new_position = self._get_new_position(action)
        if np.any(new_position < 0) or new_position[0] > self.grid_size[1] or new_position[1] > self.grid_size[0]:
             raise ValueError

        reward = 0
        done = False

        # Достигли цели
        if np.all(new_position == self.target):
            reward = 10
            done = True
        # Врезались в непроходимое препятствие
        elif self._at(new_position) == 0:
            reward = 0
            done = True
        # Превысили допустимое количество шагов
        elif self.i_step >= self.max_steps:
            reward = 0
            done = True
        elif action == self.actions.Stay:
            reward = -1
            done = False

        self.position = new_position
        self.i_step += 1
        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        cell_size = 80
        border_size = cell_size // 10
        w, h = self.grid_size[0], self.grid_size[1]
        img = np.zeros([cell_size * w, cell_size * h, 3], np.uint8)
        for i in range(h):
            cv2.line(img, (0, i * cell_size), (cell_size * w, i * cell_size), [128, 128, 128], 1)
        for i in range(w):
            cv2.line(img, (i * cell_size, 0), (i * cell_size, cell_size * h), [128, 128, 128], 1)
        for i in range(w):
            for j in range(h):
                color = tuple(np.array([self._at([i, j])] * 3).tolist())
                cv2.rectangle(img, (i * cell_size + border_size, j * cell_size + border_size),
                              ((i + 1) * cell_size - border_size, (j + 1) * cell_size - border_size),
                              color, cv2.FILLED)

        i, j = self.position
        color = (0, 0, 255)
        cv2.rectangle(img, (i * cell_size + border_size, j * cell_size + border_size),
                      ((i + 1) * cell_size - border_size, (j + 1) * cell_size - border_size),
                      color, cv2.FILLED)
        i, j = self.target
        color = (0, 255, 0)
        cv2.rectangle(img, (i * cell_size + border_size, j * cell_size + border_size),
                      ((i + 1) * cell_size - border_size, (j + 1) * cell_size - border_size),
                      color, cv2.FILLED)

        if mode == 'human':
            ar.imshow(img, resize=False)
        elif mode == 'rgb_array':
            return img
        else:
            super(GymGrid, self).render(mode)

    def _at(self, position):
        return self.grid[position[0], position[1]]

    def _random_position(self):
        return np.array([self.np_random.randint(1, self.grid_size[0] - 1),
                         self.np_random.randint(1, self.grid_size[1] - 1)], np.int64)

    def _get_observation(self):
        agent = np.zeros_like(self.grid)
        agent[self.position[1], self.position[0]] = 255
        target = np.zeros_like(self.grid)
        target[self.target[1], self.target[0]] = 255
        img = np.stack([agent, target, self.grid], -1)
        obs = {
            'image': img,
            'mission': '',
        }
        return obs

    def _get_new_position(self, action: 'GymGrid.Actions'):
        if action == GymGrid.Actions.Up:
            return self.position + [-1, 0]
        elif action == GymGrid.Actions.UpRight:
            return self.position + [-1, 1]
        elif action == GymGrid.Actions.Right:
            return self.position + [0, 1]
        elif action == GymGrid.Actions.DownRight:
            return self.position + [1, 1]
        elif action == GymGrid.Actions.Down:
            return self.position + [1, 0]
        elif action == GymGrid.Actions.DownLeft:
            return self.position + [1, -1]
        elif action == GymGrid.Actions.Left:
            return self.position + [0, -1]
        elif action == GymGrid.Actions.UpLeft:
            return self.position + [-1, -1]
        elif action == GymGrid.Actions.Stay:
            return self.position


gym.register(
    id='GymGrid-v0',
    entry_point='gym_grid:GymGrid'
)


if __name__ == '__main__':
    env = GymGrid((8, 8))
    for i in range(10000):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()