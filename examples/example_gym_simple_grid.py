import cv2
from gym_simple_grid import GymSimpleGrid


keys = {
    '1': 49,
    '2': 50,
    '3': 51,
    '4': 52,
    '5': 53,
    '6': 54,
    '7': 55,
    '8': 56,
    '9': 57,
    '0': 48,
    'q': 113,
    'esc': 27,
}


def gym_simple_grid():
    env = GymSimpleGrid((10, 10), num_obstacles=10)

    actions = {
        keys['1']: env.actions.DownLeft,
        keys['2']: env.actions.Down,
        keys['3']: env.actions.DownRight,
        keys['4']: env.actions.Left,
        keys['5']: env.actions.Stay,
        keys['6']: env.actions.Right,
        keys['7']: env.actions.UpLeft,
        keys['8']: env.actions.Up,
        keys['9']: env.actions.UpRight,
    }

    for i in range(10000):
        img = env.render('rgb_array')
        cv2.imshow("", img)
        key = cv2.waitKey(0)
        if key in (keys['q'], keys['esc']):
            exit()
        try:
            action = actions[key]
        except KeyError:
            action = env.actions.Stay
        obs, reward, done, info = env.step(action)
        if done:
            print(f"Reward: {reward}")
            env.reset()


if __name__ == '__main__':
    gym_simple_grid()
