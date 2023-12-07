import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

T_MAX = 1000
SENSOR_COV = 0.05 * np.eye(3)
NUM_PARTICLES = 1000
X_MAX = 8
Y_MAX = 4


class Particle():
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 0.0
        
    
    
class ParticleFilter():
    def __init__(self, num_particles) -> None:
        self.particles = []
        self.num_particles = num_particles
        self.random_init()

    def random_init(self):
        self.particles = np.random.rand(3, self.num_particles)
        self.particles[0] = (self.particles[0] * 2 - 1) * X_MAX
        self.particles[1] = (self.particles[1] * 2 - 1) * Y_MAX
        self.particles[2] = (self.particles[2] * 2 - 1) * np.pi
        self.particles = self.particles.T


    def action_model(self):
        pass
    
    def sensor_model(self):
        pass
    
    def low_var_resample(self):
        pass
    
def get_action(path: np.ndarray, t) -> np.ndarray:
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    u_t: action the robot make for next step (configuration difference)
    '''
    return path[t] - path[t - 1]


def get_sensor(path: np.ndarray, t) -> np.ndarray:
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    z_t: sensor reading
    '''
    true_config = path[t]
    noisey_config = np.random.multivariate_normal(true_config, SENSOR_COV)
    return noisey_config

    
    
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
    x_after_interpolate = np.linspace(0, path.shape[1], T_MAX)
    path_temp = []
    for item in path:
        path_temp.append(np.interp(x_after_interpolate, x_before_interpolate, np.squeeze(item)))
    path = np.array(path_temp).T

    # particle filter
    t = 0
    u_cache = []
    z_cache = []
    pf = ParticleFilter(NUM_PARTICLES)
    while(t < T_MAX - 1):  
        t += 1
        u_t = get_action(path, t)
        u_cache.append(u_t)
        z_t = get_sensor(path, t)
        z_cache.append(z_t)


    # plotting
    plt.figure(1)
    plt.scatter(path.T[0], path.T[1], s=5)
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=5)

    plt.figure(2)
    plt.scatter(pf.particles.T[0], pf.particles.T[1], s=5)

    plt.show()



if __name__ == '__main__':
    main()
    
    