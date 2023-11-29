from pf import *
from ekf import *
import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time


def main(screenshot=False):

     # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2maze.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))

    wait_for_user()
    print(f"==================\nDemo running...\nSome notes comes here\n==================")
    wait_for_user()

    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))


    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()