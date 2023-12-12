from pf import *
from ekf import *
import numpy as np
import pybullet as p
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time


def main(screenshot=False):

     # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    
    ############### Change map here ###############
    robots, obstacles = load_env('pr2maze.json')
    # robots, obstacles = load_env('pr2empty.json')
    # obots, obstacles = load_env('pr2complexMaze.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    wait_for_user()
    print(f"==================\nDemo running...\nSome notes comes here\n==================")
    wait_for_user()

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))

    # read path
    path = []
    line_temp = []
    with open('path_maze.txt', 'r') as file:
        for line in file:
            if ']' in line:
                line_temp.append(line)
                joint_line = ''.join(line_temp).replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                joint_line = np.array([float(num) for num in joint_line if num is not ''])
                path.append(joint_line)
                line_temp = []
            else:
                line_temp.append(line)
    path = np.array(path)

    # interpolate path
    x_before_interpolate = np.linspace(0, path.shape[1] - 1, path.shape[1])
    x_after_interpolate = np.linspace(0, path.shape[1], 1000)
    path_temp = []
    for item in path:
        path_temp.append(np.interp(x_after_interpolate, x_before_interpolate, np.squeeze(item)))
    path = np.array(path_temp).T


    # execute path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.001)

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()