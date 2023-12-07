import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as scistats
import copy

T_MAX = 1000 # iteration for particle filter & len of interpolated path
SENSOR_STD = [0.05, 0.05, 0.05] #[x, y, theta]
ACTION_STD = [0.05, 0.05, 0.05] #[x, y, theta]
NUM_PARTICLES = 1000 # number of particles for particle filter
X_MAX = 8 # max x length for map
Y_MAX = 4 # max y length for map
    
class ParticleFilter():
    def __init__(self, num_particles, x_max, y_max) -> None:
        self.num_particles = num_particles
        self.particles_t0 = []
        self.particles_tminus1 = []
        self.random_init(x_max, y_max)
        self.particles_t = []
        self.samples_t = []
        self.weight_t = []
        self.u_t = []
        self.z_t = []

    def random_init(self, x_max, y_max):
        self.particles_t0 = np.random.rand(3, self.num_particles)
        self.particles_t0[0] = (self.particles_t0[0] * 2 - 1) * x_max
        self.particles_t0[1] = (self.particles_t0[1] * 2 - 1) * y_max
        self.particles_t0[2] = (self.particles_t0[2] * 2 - 1) * np.pi
        self.particles_t0 = self.particles_t0.T
        self.particles_tminus1 = self.particles_t0

    def action_model(self, action_std):
        '''
        Sample x_t_m ~ p(x_t | u_t, x_tminus1)
        Update self.particle_t
        '''
        mean = self.u_t
        # cov = np.eye(3)
        # cov[0, 0] = np.sqrt(action_std[0] * mean[0])
        # cov[1, 1] = np.sqrt(action_std[1] * mean[1])
        # cov[2, 2] = np.sqrt(action_std[2] * mean[2])
        # actual_dxytheta = np.random.multivariate_normal(mean, cov)
        cov_x = np.sqrt(action_std[0] * mean[0])
        cov_y = np.sqrt(action_std[1] * mean[1])
        cov_theta = np.sqrt(action_std[2] * mean[2])
        
        actual_dx = np.random.normal(mean[0], cov_x)
        actual_dy = np.random.normal(mean[1], cov_y)
        actual_dtheta = np.random.normal(mean[2], cov_theta)
        actual_dxytheta = np.array([actual_dx, actual_dy, actual_dtheta])
        
        particles_t = self.particles_tminus1 + actual_dxytheta
        particles_t.T[2] = warp_to_pi(particles_t.T[2])
        self.particles_t = particles_t
    
    def sensor_model(self, sensor_std):
        '''
        w_t_m = p(z_t | x_t_m)
        S_t = S_t U (x_t_m, w_t_m)
        Update self.weight_t and self.sample_t
        '''
        w_sum = 0
        self.weight_t = []
        
        sensor_cov = np.eye(3)
        sensor_cov[0,0] = sensor_std[0]
        sensor_cov[1,1] = sensor_std[1]
        sensor_cov[2,2] = sensor_std[2]
        p = scistats.multivariate_normal(self.z_t, sensor_cov)
        self.weight_t = np.array(self.weight_t)
        for particle in self.particles_t:
            w_t_m = p.pdf(list(particle))
            self.weight_t = np.append(self.weight_t, w_t_m)
            w_sum += w_t_m
        self.weight_t /= w_sum
        # print(self.weight_t.reshape(-1, 1).shape)
        self.samples_t = np.concatenate((self.particles_t, self.weight_t.reshape(-1, 1)), axis=1)
        
        
    
    def low_var_resample(self):
        '''
        Update self.particle_tminus1
        '''
        
        
    
def get_action(path: np.ndarray, t) -> np.ndarray:
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    u_t: action the robot make for next step (configuration difference)
    '''
    return path[t] - path[t - 1]


def get_sensor(path: np.ndarray, t, sensor_std) -> np.ndarray: # maybe can randomly generate a config within the map and plugin
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    z_t: sensor reading
    '''
    true_config = path[t]
    sensor_cov = np.eye(3)
    sensor_cov[0,0] = sensor_std[0]
    sensor_cov[1,1] = sensor_std[1]
    sensor_cov[2,2] = sensor_std[2]
    
    noisey_config = np.random.multivariate_normal(true_config, sensor_cov)
    return noisey_config

def warp_to_pi(angles):
    angles = np.mod(angles, 2 * np.pi)
    angles[angles > np.pi] -= 2 * np.pi
    return angles
    
def main():
    ################ read path ################
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

    ################ particle filter ################
    t = 0
    u_cache = []
    z_cache = []
    pf = ParticleFilter(NUM_PARTICLES, X_MAX, Y_MAX)
    
    while(t < T_MAX - 1):  
        t += 1
        print(f"Num of iteration: {t}")
        
        # get control input and sensor data
        pf.u_t = get_action(path, t)
        u_cache.append(pf.u_t)
        pf.z_t = get_sensor(path, t, SENSOR_STD)
        z_cache.append(pf.z_t)
        
        # reset parameter
        pf.samples_t = []
        pf.particles_t = []
        
        # apply action model
        pf.action_model(ACTION_STD)
        
        # apply sensor model
        pf.sensor_model(SENSOR_STD)
        
        # apply resampling
        pf.low_var_resample()
        
            


    ################ plotting ################
    plt.figure(1)
    plt.scatter(path.T[0], path.T[1], s=5)
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=5)

    plt.figure(2)
    plt.scatter(pf.particles_t0.T[0], pf.particles_t0.T[1], s=5)
    
    plt.figure(3)
    plt.scatter(pf.particles_t.T[0], pf.particles_t.T[1], s=5)

    plt.show()



if __name__ == '__main__':
    main()
    
    