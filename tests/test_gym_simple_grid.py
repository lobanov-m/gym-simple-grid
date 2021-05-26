from gym_simple_grid import GymSimpleGrid


def check_raises_exception(func, exception):
    try:
        func()
    except exception:
        pass
    else:
        raise ValueError


def test_gym_simple_grid():
    check_raises_exception(lambda: GymSimpleGrid((10, 10), (-1, -1)), AssertionError)
    check_raises_exception(lambda: GymSimpleGrid((30, 10), (0, 30)), AssertionError)
    check_raises_exception(lambda: GymSimpleGrid((10, 10), (0, 0), (0, 0)), AssertionError)
    check_raises_exception(lambda: GymSimpleGrid((10, 10), (0, 0), (10, 10)), AssertionError)
    check_raises_exception(lambda: GymSimpleGrid((10, 10), (0, 0), obstacles=[(0, 0)]), AssertionError)
    check_raises_exception(lambda: GymSimpleGrid((10, 10), obstacles=[], num_obstacles=2), AssertionError)


if __name__ == '__main__':
    test_gym_simple_grid()
