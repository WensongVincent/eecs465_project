import numpy as np
import numpy.linalg as la
import re

class Particle():
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0
        
    
    
class ParticleFilter():
    def __init__(self) -> None:
        self.particles = []
        self.num_particles = len(self.particles)

    
    def action_model():
        pass
    
    def sensor_model():
        pass
    
    def low_var_resample():
        pass
    
def get_action():
    pass

def get_sensor():
    pass
    
def main():
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

    t = 0
    pf = ParticleFilter()
    while(True):
        t += 1
        u_t = get_action()
        

if __name__ == '__main__':
    main()
    
    