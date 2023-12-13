import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B_func, C, R, Q, initial_state, initial_covariance):
        self.A = A
        self.B_func = B_func
        self.C = C
        self.R = R
        self.Q = Q
        self.state = initial_state
        self.covariance = initial_covariance
 
    def predict(self, control_input):
        theta = self.state[2]
        self.B = self.B_func(theta)
        self.state = np.dot(self.A, self.state) + np.dot(self.B, control_input)
        self.covariance = np.dot(np.dot(self.A, self.covariance), self.A.T) + self.Q

    def update(self, measurement):   
        K = np.dot(np.dot(self.covariance, self.C.T), 
                   np.linalg.inv(np.dot(np.dot(self.C, self.covariance), self.C.T) + self.R))
        self.state = self.state + np.dot(K, (measurement - np.dot(self.C, self.state)))
        self.covariance = self.covariance - np.dot(np.dot(K, self.C), self.covariance)

    def get_state(self):
        return self.state

# -- Read data from file
def read_path_from_file(file_path):
    path = []
    line_temp = []
    with open(file_path, 'r') as file:
        for line in file:
            if ']' in line:
                line_temp.append(line)
                joint_line = ''.join(line_temp).replace('[', ' ').replace(']', ' ').replace('\n', ' ').split(' ')
                joint_line = np.array([float(num) for num in joint_line if num != ''])
                path.append(joint_line)
                line_temp = []
            else:
                line_temp.append(line)
    path = np.array(path)

    x_before_interpolate = np.linspace(0, path.shape[1] - 1, path.shape[1])
    x_after_interpolate = np.linspace(0, path.shape[1], 200) # extend data point into 300
    path_temp = []
    for item in path:
        path_temp.append(np.interp(x_after_interpolate, x_before_interpolate, np.squeeze(item)))

    return np.array(path_temp) 

# -- Simulate a location sensor with Guassian noise
def location_sensor_measurements(true_state, sensor_noise_covariance):
    measured_positions = np.zeros_like(true_state)
    for i in range(true_state.shape[1]):
        x_true, y_true, theta_true = true_state[:, i]
        
        # Generate noise from multivariate normal distribution
        noise = np.random.multivariate_normal([0, 0, 0], sensor_noise_covariance)
        measured_positions[:, i] = [x_true, y_true, theta_true] + noise
    return measured_positions

# -- Calculate error in rmse
def calculate_rmse(estimated_states, true_states):
    if estimated_states.shape != true_states.shape:
        raise ValueError("The shapes of the estimated and true states must be the same.")
    
    squared_errors = (estimated_states - true_states) ** 2
    mean_squared_errors = squared_errors.mean(axis=0)
    rmse = np.sqrt(mean_squared_errors)
    return rmse

def main():
    # Read path from recorded data 
    true_state = read_path_from_file('path_empty.txt')
    x_true, y_true, theta_true = true_state

    ################ Kalman Filter ################
    # KF initialization
    from pr2_models import A, B, C, R, Q
    initial_state = true_state[:,0]  # x_0: x, y, θ
    initial_covariance = np.eye(3)  # sigma_0: initial covarience
    kf = KalmanFilter(A, B, C, R, Q, initial_state, initial_covariance)

    # Generate location sensor measurements
    measured_state = location_sensor_measurements(true_state, Q)
    x_measured, y_measured, theta_measured = measured_state

    # Estimate the state of a pr2 robot
    kf_states = []
    kf_states.append(initial_state)
    
    for i in range(1, true_state.shape[1]):
        # control input
        dx = x_true[i] - x_true[i-1]
        dy = y_true[i] - y_true[i-1]
        dtheta = theta_true[i] - theta_true[i-1]
        control_input = np.array([dx, dy, dtheta])

        # prediction step
        kf.predict(control_input) 

        # correction step
        kf.update(measured_state[:, i])  
        kf_states.append(kf.get_state())

    kf_states = np.array(kf_states)

    # Calculate error
    rmse = calculate_rmse(kf_states, true_state.T)
    print("KF RMSE: ", rmse)

    ################ Visualization ################
    # Set the figure size
    plt.figure(figsize=(8.5, 6))
    plt.xlim(-4,4)
    plt.ylim(-2,2)

    # Plotting the actual, measured, and KF paths
    plt.plot(x_true, y_true, 'b-', label="Ground Truth", linewidth=2) 
    plt.scatter(x_measured, y_measured, color='g', s=30, label="Sensor Data", alpha=0.5)  
    # plt.plot(kf_states[:, 0], kf_states[:, 1], 'r--', label="KF estimation", linewidth=2)  
    plt.scatter(kf_states[:, 0], kf_states[:, 1], color='r', s=30, label="KF estimation", alpha=0.5) 

    # Add arrows to show orientation at selected points
    arrow_skip = 10 # Number of points to skip between arrows
    for i in range(0, len(theta_true), arrow_skip):
        plt.arrow(x_true[i], y_true[i], 
                  0.1 * np.cos(theta_true[i]), 0.1 * np.sin(theta_true[i]), 
                  head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        
    # Add arrows to show orientation for KF path
    for i in range(0, kf_states.shape[0], arrow_skip):
        plt.arrow(kf_states[i, 0], kf_states[i, 1], 
                  0.1 * np.cos(kf_states[i, 2]), 0.1 * np.sin(kf_states[i, 2]), 
                  head_width=0.05, head_length=0.1, fc='purple', ec='purple')

    # Marking start and end points for each path
    plt.scatter(x_true[0], y_true[0], color='b', marker='o', s=100, label="Start (Actual)", edgecolor='black')
    plt.scatter(x_true[-1], y_true[-1], color='b', marker='X', s=100, label="End (Actual)", edgecolor='black')
    plt.scatter(kf_states[0, 0], kf_states[0, 1], color='r', marker='o', s=100, label="Start (EKF)", edgecolor='black')
    plt.scatter(kf_states[-1, 0], kf_states[-1, 1], color='r', marker='X', s=100, label="End (EKF)", edgecolor='black')

    # Adding labels, title, grid, and legend
    plt.xlabel("X Position") 
    plt.ylabel("Y Position") 
    plt.title("Kalman Filter Path Tracking") 
    plt.legend() 
    plt.grid(True) 
    plt.show()

if __name__ == '__main__':
    main()
