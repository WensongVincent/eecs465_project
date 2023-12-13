import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as scistats
import copy

from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, wait_for_user,get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS

T_MAX = 300 # iteration for particle filter & len of interpolated path
ACTION_COV = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]) #[0.05, 0.05, 0.05] #[x, y, theta]
SENSOR_COV = np.array([[0.02, 0.001, 0], [0.001, 0.02, 0], [0, 0, 0.02]]) #[0.03, 0.03, 0.03] #[x, y, theta]
NUM_PARTICLES = 1000 # number of particles for particle filter
X_MAX = 8 # max x length for map
Y_MAX = 4 # max y length for map
ACTION_ONLY = False
    
class ParticleFilter():
    def __init__(self, num_particles, x_max, y_max, collision_fn) -> None:
        self.num_particles = num_particles
        self.particles_t0 = []
        self.particles_tminus1 = []
        self.random_init(x_max, y_max, collision_fn)
        self.particles_t = []
        self.samples_t = [] #[[x, y, theta, w],...]
        self.weight_t = []
        self.weight_tminus1 = []
        self.u_t = []
        self.z_t = []
        self.estimated_path = []

    def random_init(self, x_max, y_max, collision_fn):
        # self.particles_t0 = np.random.rand(3, self.num_particles)
        # self.particles_t0[0] = (self.particles_t0[0] * 2 - 1) * x_max
        # self.particles_t0[1] = (self.particles_t0[1] * 2 - 1) * y_max
        # self.particles_t0[2] = (self.particles_t0[2] * 2 - 1) * np.pi
        # self.particles_t0 = self.particles_t0.T
        # self.particles_tminus1 = self.particles_t0

        particles = []
        while len(particles) < self.num_particles:
            particle_x = np.random.uniform(-x_max, x_max, 1)
            particle_y = np.random.uniform(-y_max, y_max, 1)
            particle_theta = np.random.uniform(-np.pi, np.pi, 1)
            if not collision_fn([particle_x, particle_y, particle_theta]):
                particles.append(np.array([particle_x, particle_y, particle_theta]))
        
        self.particles_t0 = np.array(particles).squeeze()
        self.particles_tminus1 = np.array(particles).squeeze()

    def action_model(self, action_cov):
        '''
        Sample x_t_m ~ p(x_t | u_t, x_tminus1)
        Update self.particle_t
        '''
        delta = 7e-5
        mean = self.u_t
        cov_x = np.sqrt(action_cov[0, 0] * np.abs(mean[0]) + delta)
        cov_y = np.sqrt(action_cov[1, 1] * np.abs(mean[1]) + delta)
        cov_theta = np.sqrt(action_cov[2, 2] * np.abs(mean[2]) + delta)
        
        actual_dx = np.random.normal(mean[0], cov_x, self.num_particles).reshape(self.num_particles, -1)
        actual_dy = np.random.normal(mean[1], cov_y, self.num_particles).reshape(self.num_particles, -1)
        actual_dtheta = np.random.normal(mean[2], cov_theta, self.num_particles).reshape(self.num_particles, -1)
        actual_dxytheta = np.concatenate((actual_dx, actual_dy, actual_dtheta), axis=1)
        
        particles_t = self.particles_tminus1 + actual_dxytheta
        particles_t.T[2] = warp_to_pi(particles_t.T[2])
        self.particles_t = particles_t
    
    def sensor_model(self, sensor_cov, action_only):
        '''
        w_t_m = p(z_t | x_t_m)
        S_t = S_t U (x_t_m, w_t_m)
        Update self.weight_t
        '''
        w_sum = 0
        self.weight_t = []
        
        # cov = np.eye(3)
        # cov[0,0] = sensor_cov[0]
        # cov[1,1] = sensor_cov[1]
        # cov[2,2] = sensor_cov[2]
        cov = sensor_cov
        p = scistats.multivariate_normal(self.z_t, cov)
        self.weight_t = np.array(self.weight_t)
        for particle in self.particles_t:
            w_t_m = p.pdf(list(particle))
            self.weight_t = np.append(self.weight_t, w_t_m)
            w_sum += w_t_m
        self.weight_t /= w_sum
        # print(self.weight_t.reshape(-1, 1).shape)
        # self.samples_t = np.concatenate((self.particles_t, self.weight_t.reshape(-1, 1)), axis=1)
        
        # self.estimated_path.append(self.particles_t[self.weight_t.argmax(), :])
        
        
    def low_var_resample(self, action_only):
        '''
        Update self.samplt_t, self.particle_tminus1, self.weight_tminus1
        '''
        self.particles_tminus1 = np.zeros((self.num_particles, 3))
        self.samples_t = np.zeros((self.num_particles, 4))
        
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.weight_t[0]
        i = 0
        for m in range(self.num_particles):
            U = r + m * (1 / self.num_particles)
            while U > c and i < self.num_particles - 1:
                i += 1
                c += self.weight_t[i]
            # self.particles_tminus1[m] = self.particles_t[i, :]
            # import pdb; pdb.set_trace()
            self.samples_t[m,:] = copy.deepcopy(np.concatenate((self.particles_t[i, :].reshape(1,-1), self.weight_t[i].reshape(1,-1)), axis=1))
        self.samples_t[:, 3] /= self.samples_t[:, 3].sum()
        self.particles_tminus1 = copy.deepcopy(self.samples_t[:,:3])
        self.weight_tminus1 = copy.deepcopy(self.samples_t[:,3])
        
    
    def estimate_config(self):
        x = (self.samples_t[:, 0] * self.samples_t[:, 3]).sum()
        y = (self.samples_t[:, 1] * self.samples_t[:, 3]).sum()
        theta_cos = (np.cos(self.samples_t[:, 2]) * self.samples_t[:, 3]).sum()
        theta_sin = (np.sin(self.samples_t[:, 2]) * self.samples_t[:, 3]).sum()
        theta = np.arctan2(theta_cos, theta_sin)
        
        self.estimated_path.append(np.array([x, y, theta]))
        
    
def get_action(path: np.ndarray, t) -> np.ndarray:
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    u_t: action the robot make for next step (configuration difference)
    '''
    moved = False
    u_t = path[t] - path[t - 1]
    if la.norm(u_t) > 1e-10:
        moved = True
    return u_t, moved

def get_sensor(path: np.ndarray, t, sensor_cov) -> np.ndarray: # maybe can randomly generate a config within the map and plugin
    '''
    Input: 
    path: path given, shape:(M, 3)
    t: current time step

    Output: 
    z_t: sensor reading
    '''
    measured = True # set to always true for now since always taking sensor measurement
    true_config = path[t]
    # cov = np.eye(3)
    # cov[0,0] = sensor_cov[0]
    # cov[1,1] = sensor_cov[1]
    # cov[2,2] = sensor_cov[2]
    cov = sensor_cov
    
    noisey_config = np.random.multivariate_normal(true_config, cov)
    return noisey_config, measured

def warp_to_pi(angles):
    angles = np.mod(angles, 2 * np.pi)
    angles[angles > np.pi] -= 2 * np.pi
    return angles
    
def main():
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    
    ############### Change map here ###############
    # robots, obstacles = load_env('pr2maze.json')
    # robots, obstacles = load_env('pr2empty.json')
    robots, obstacles = load_env('pr2complexMaze.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))


    ################ read path ################
    path = []
    line_temp = []
    # with open('path_maze.txt', 'r') as file:
    # with open('path_empty.txt', 'r') as file:
    with open('path_complexMaze.txt', 'r') as file:
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
    # print(path.shape)
    
    
    ################ particle filter ################
    t = 0
    u_cache = []
    z_cache = []
    particles_cache = []
    pf = ParticleFilter(NUM_PARTICLES, X_MAX, Y_MAX, collision_fn)
    moved = False
    measured = False
    
    while(t < T_MAX - 1):  
        t += 1
        
        # get control input and sensor data
        pf.u_t, moved = get_action(path, t)
        u_cache.append(pf.u_t)
        # if not moved:
        #     print(f"Not moved at step {t}")
        pf.z_t, measured = get_sensor(path, t, SENSOR_COV)
        z_cache.append(pf.z_t)
        
        if moved and measured:
            # reset parameter
            pf.samples_t = []
            pf.particles_t = []
            
            # apply action model
            pf.action_model(ACTION_COV)
            # plt.figure(1)
            # plt.scatter(pf.particles_t.T[0], pf.particles_t.T[1], s=5)
            # plt.show()
            
            # apply sensor model
            pf.sensor_model(SENSOR_COV, ACTION_ONLY)
            
            # apply resampling
            pf.low_var_resample(ACTION_ONLY)
            # plt.figure(2)
            # plt.scatter(pf.particles_tminus1.T[0], pf.particles_tminus1.T[1], s=5)
            # plt.show
            
            pf.estimate_config()
            
            if t == 0 or t%30 == 0 or t == T_MAX-1:
                particles_cache.append(pf.particles_t)
                print(f"Num of iteration: {t}/{T_MAX}")

    ################ plotting ################
    plt.figure(1)
    plt.plot(path.T[0], path.T[1], label='Desired Path', c='b')
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=5, label='Sensor Measurement', c='g')
    # plt.plot(np.array(pf.estimated_path).T[0], np.array(pf.estimated_path).T[1], linestyle = '--',label='Estimated Path')
    plt.scatter(np.array(pf.estimated_path).T[0], np.array(pf.estimated_path).T[1], s=5 ,label='Estimated Path', c='r')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title('PF desired path, sensor measurement and estimated path')
    
    plt.figure(2)
    plt.plot(path.T[0], path.T[1], label='Desired Path', c='b')
    plt.scatter(np.array(z_cache).T[0], np.array(z_cache).T[1], s=5, label='Sensor Measurement', c='g')
    plt.scatter(np.array(particles_cache).T[0], np.array(particles_cache).T[1], s=5 ,label='Particles example', c='r')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title('PF desired path, sensor measurement and particle examples')

    plt.figure(3)
    plt.scatter(pf.particles_t0.T[0], pf.particles_t0.T[1], s=5, c='r')

    plt.show()


if __name__ == '__main__':
    main()
    
    