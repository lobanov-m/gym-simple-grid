import cv2
import gym
import gym.utils.seeding
import numpy as np

from enum import IntEnum

from astar import AStar

from typing import Optional, Tuple, List


class AStarForGymSimpleGrid(AStar):
    def __init__(self, grid):
        self.grid = grid

    def neighbors(self, node):
        neighbours_offsets = [
            [-1, -1],
            [ 0, -1],
            [ 1, -1],
            [ 1,  0],
            [ 1,  1],
            [ 0,  1],
            [-1,  1],
            [-1,  0],
        ]
        neighbours = [(node[0] + offset[0], node[1] + offset[1]) for offset in neighbours_offsets]
        return neighbours

    def distance_between(self, n1, n2):
        d = np.linalg.norm(np.array(n1) - np.array(n2))
        h, w = self.grid.shape
        if n2[0] >= w or n2[1] >= h or self.grid[n2[1], n2[0]] == 0:
            d = 255
        return d

    def heuristic_cost_estimate(self, current, goal):
        return np.linalg.norm(np.array(goal) - current)

    def is_goal_reached(self, current, goal):
        return current == goal


class GymSimpleGrid(gym.Env):
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

    def __init__(self, grid_size: Tuple[int, int] = (20, 20),
                 start_position: Tuple[int, int] = None,
                 target: Tuple[int, int] = None,
                 obstacles: List[Tuple[int, int]] = None,
                 num_obstacles: int = None,
                 max_steps: int = 100,
                 seed: int = 1,
                 target_image_size=400):
        """
        Args:
            grid_size: Size of gym field, (w, h)
            start_position: Initial position of agent in grid. If None, then randomized.
                            (x, y), 0 < x < grid_size[0], 0 < y < grid_size[1]
            target: Position of target in grid. If None, then randomized.
                    (x, y), 0 < x < grid_size[0], 0 < y < grid_size[1]
            obstacles: Optional list of obstacles coordinates. If None then `num_obstacles` obstacles will be generated 
            num_obstacles: Optional number of obstacles. Used only if `obstacles` if None
            max_steps: Maximum number of steps after which gym will be restarted
            seed: Random generator seed
            target_image_size: Size of the biggest side of rendered image
        """
        # Size of grid with borders
        self._grid_size = (grid_size[0] + 2, grid_size[1] + 2)

        if start_position is not None:
            assert 0 <= start_position[0] < self.grid_size[0] and 0 <= start_position[1] < self.grid_size[1]
        self.start_position = start_position

        if target is not None:
            assert target != start_position
            assert 0 <= target[0] < self.grid_size[0] and 0 <= target[1] < self.grid_size[1]
        self.target = target

        self.obstacles = obstacles
        self.num_obstacles = 0
        if obstacles is not None:
            assert num_obstacles is None, "If use `obstacles`, `num_obstacles` should be None"
            for x, y in obstacles:
                assert (x, y) != start_position and (x, y) != target
                assert 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
            self.num_obstacles = len(obstacles)
        elif num_obstacles is not None:
            self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.target_image_size = target_image_size

        self.actions = GymSimpleGrid.Actions
        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.grid = None
        self.position = None
        self._target = None
        self.done = None
        self.history = None
        self.astar = None
        self.astar_path = None
        self.astar_distance = None
        self.i_step = None

        image_observation_space = gym.spaces.Box(low=0, high=255, shape=(self._grid_size[1], self._grid_size[0], 3),
                                                 dtype='uint8')
        self.observation_space = gym.spaces.Dict({'image': image_observation_space})

        self.np_random = None
        self.seed(seed=seed)
        self.reset()

    @property
    def grid_size(self):
        return self._grid_size[0] - 2, self._grid_size[1] - 2

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.grid = np.ones((self._grid_size[1], self._grid_size[0]), np.uint8) * 255
        self.grid[0] = 0
        self.grid[-1] = 0
        self.grid[:, 0] = 0
        self.grid[:, -1] = 0

        if self.start_position is None:
            self.position = self._random_position()
        else:
            self.position = np.array([self.start_position[0] + 1, self.start_position[1] + 1])
        
        if self.target is None:
            while 1:
                self._target = self._random_position()
                if np.any(self._target != self.position):
                    break
        else:
            self._target = np.array([self.target[0] + 1, self.target[1] + 1])
        self.grid[self._target[1], self._target[0]] = 255

        if self.obstacles is None:
            num_obstacles = 0
            while num_obstacles != self.num_obstacles:
                x0 = self.np_random.randint(1, self._grid_size[0] - 1)
                y0 = self.np_random.randint(1, self._grid_size[1] - 1)
                if np.all(np.array([x0, y0]) == self.position) or np.all(np.array([x0, y0]) == self.target):
                    continue
                self.grid[y0, x0] = 0
                num_obstacles += 1
        else:
            for x0, y0 in self.obstacles:
                x0 += 1
                y0 += 1
                if np.all(np.array([x0, y0]) == self.position) or np.all(np.array([x0, y0]) == self.target):
                    continue
                self.grid[y0, x0] = 0

        self.done = False
        self.history = [self.position]
        self.astar = AStarForGymSimpleGrid(self.grid)
        self.astar_path = list(self.astar.astar(tuple(self.position.tolist()), tuple(self._target.tolist())))
        self.astar_distance = self._calculate_distance(self.astar_path)
        self.i_step = 0

        obs = self._get_observation()
        return obs

    def step(self, action):
        self.i_step += 1
        assert self._at(self.position) != 0, f'{self.history}'
        new_position = self._get_new_position(action)
        if np.any(new_position < 0) or new_position[0] > self._grid_size[0] or new_position[1] > self._grid_size[1]:
            raise ValueError
        self.position = new_position
        self.history.append(new_position)

        reward = 0
        done = False

        # Достигли цели
        if np.all(new_position == self._target):
            d = self._calculate_distance(self.history)  # Расчитываем длину пройденного пути
            reward = 10 * self.astar_distance / d if d != 0 else 1
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
        self.done = done

        obs = self._get_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        cell_size = self.target_image_size // max(self.grid_size)
        border_size = cell_size // 10 if cell_size > 10 else 1
        w, h = self._grid_size[0], self._grid_size[1]
        img = np.zeros([cell_size * h, cell_size * w, 3], np.uint8)
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
        i, j = self._target
        color = (0, 255, 0)
        cv2.rectangle(img, (i * cell_size + border_size, j * cell_size + border_size),
                      ((i + 1) * cell_size - border_size, (j + 1) * cell_size - border_size),
                      color, cv2.FILLED)
        for i in range(len(self.history) - 1):
            x0, y0 = self.history[i][0] * cell_size + cell_size // 2, self.history[i][1] * cell_size + cell_size // 2
            x1 = self.history[i + 1][0] * cell_size + cell_size // 2
            y1 = self.history[i + 1][1] * cell_size + cell_size // 2
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255))

        for i in range(len(self.astar_path) - 1):
            x0 = self.astar_path[i][0] * cell_size + cell_size // 2
            y0 = self.astar_path[i][1] * cell_size + cell_size // 2
            x1 = self.astar_path[i + 1][0] * cell_size + cell_size // 2
            y1 = self.astar_path[i + 1][1] * cell_size + cell_size // 2
            cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0))

        if mode == 'human':
            cv2.imshow("", img)
            cv2.waitKey()
        elif mode == 'rgb_array':
            return img
        else:
            super(GymSimpleGrid, self).render(mode)

    def _at(self, position):
        return self.grid[position[1], position[0]]

    def _random_position(self):
        return np.array([self.np_random.randint(1, self._grid_size[0] - 1),
                         self.np_random.randint(1, self._grid_size[1] - 1)], np.int64)

    def _get_observation(self):
        agent = np.zeros_like(self.grid)
        agent[self.position[1], self.position[0]] = 255
        target = np.zeros_like(self.grid)
        target[self._target[1], self._target[0]] = 255
        img = np.stack([agent, target, self.grid], -1)
        obs = {
            'image': img,
            'mission': '',
        }
        return obs

    def _get_new_position(self, action: 'GymGrid.Actions'):
        if action == GymSimpleGrid.Actions.Up:
            return self.position + [0, -1]
        elif action == GymSimpleGrid.Actions.UpRight:
            return self.position + [1, -1]
        elif action == GymSimpleGrid.Actions.Right:
            return self.position + [1, 0]
        elif action == GymSimpleGrid.Actions.DownRight:
            return self.position + [1, 1]
        elif action == GymSimpleGrid.Actions.Down:
            return self.position + [0, 1]
        elif action == GymSimpleGrid.Actions.DownLeft:
            return self.position + [-1, 1]
        elif action == GymSimpleGrid.Actions.Left:
            return self.position + [-1, 0]
        elif action == GymSimpleGrid.Actions.UpLeft:
            return self.position + [-1, -1]
        elif action == GymSimpleGrid.Actions.Stay:
            return self.position

    def _calculate_distance(self, path):
        d = 0
        for i in range(len(path) - 1):
            p0 = path[i]
            p1 = path[i + 1]
            d += self._dist(p0, p1)
        return d

    @staticmethod
    def _dist(p0, p1):
        return np.linalg.norm(np.array(p0) - np.array(p1))


gym.register(
    id='GymSimpleGrid-v0',
    entry_point='gym_simple_grid:GymSimpleGrid'
)
