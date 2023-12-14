from pf import *
from kf import *
from pr2_models import *
from utils_filter import *
import numpy as np
import pybullet as p
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
import copy
from tqdm import tqdm

def main(screenshot=False):
    map_list = ["pr2empty.json", "pr2maze.json" , "pr2complexMaze.json"]
    path_list = ["path_empty.txt", "path_maze.txt", "path_complexMaze.txt"]

    print(f"====================================\nDemo running...\nThere are 3 different maps in demo: pr2empty, pr2maze, pr2complexMaze\n============================")
    wait_for_user()

    for map_name, path_name in zip(map_list, path_list):

        print(f"============================\nRunning Demo with Map: {map_name}...\nShowing path...")
        # show path
        connect(use_gui=True)
        p.resetDebugVisualizerCamera(cameraDistance = 5, cameraYaw = 0, cameraPitch = -60, cameraTargetPosition = [0, 0, 0])
        robots, obstacles = load_env(map_name)
        base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]
        path_gui = read_path_from_file_no_interpolate(path_name)
        for pos in path_gui.T:
            draw(pos,'black', radius=0.07)
        execute_trajectory(robots['pr2'], base_joints, path_gui.T, sleep=0.01)
        print(f"Path shown, please following the guidance:")
        wait_if_gui()
        disconnect()

        # run kf and pf
        print(f"========\nRunning Kalman Filter...")
        main_kf(path_name, map_name)
        input("Press Enter to continue:")
        plt.close()
        
        print(f"========\nRunning Particle Filter...")
        main_pf(path_name, map_name)
        input("Press enter to continue:")
        plt.close()
        
        print(f"Demo with Map: {map_name} Finished!\n============================")


    wait_if_gui()
    
    print(f"All Demo Finished!\n====================================")
    

if __name__ == '__main__':
    main()