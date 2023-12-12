import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from kf import read_path_from_file
from kf import location_sensor_measurements

class ExtendKalmanFilter:
    def __init__(self, A, B, C, R, Q, initial_state, initial_covariance):
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.state = initial_state
        self.covariance = initial_covariance
 
    def predict(self, control_input):
        self.state = np.dot(self.A, self.state) + np.dot(self.B, control_input)
        self.covariance = np.dot(np.dot(self.A, self.covariance), self.A.T) + self.Q

    def update(self, measurement):   
        K = np.dot(np.dot(self.covariance, self.C.T), 
                   np.linalg.inv(np.dot(np.dot(self.C, self.covariance), self.C.T) + self.R))
        self.state = self.state + np.dot(K, (measurement - np.dot(self.C, self.state)))
        self.covariance = self.covariance - np.dot(np.dot(K, self.C), self.covariance)

    def get_state(self):
        return self.state

def main():
    # Read path from recorded data 
    true_state = read_path_from_file('path_maze.txt')
    x_true, y_true, theta_true = true_state

    # KF initialization
    from pr2_models import A, B, C, R, Q
    initial_state = true_state[:,0]  # x_0: x, y, θ
    initial_covariance = np.eye(3)  # sigma_0: initial covarience
    ekf = ExtendKalmanFilter(A, B, C, R, Q, initial_state, initial_covariance)

    # Generate location sensor measurements
    measured_state = location_sensor_measurements(true_state, Q)
    x_measured, y_measured, theta_measured = measured_state

    # Estimate the state of a pr2 robot
    ekf_states = []
    for i in range(1, true_state.shape[1]):
        # control input
        dx = x_true[i] - x_true[i-1]
        dy = y_true[i] - y_true[i-1]
        dtheta = theta_true[i] - theta_true[i-1]
        control_input = np.array([dx, dy, dtheta])

        ekf.predict(control_input) # prediction step
        ekf.update(measured_state[:, i]) # correction step 
        ekf_states.append(ekf.get_state())

    ekf_states = np.array(ekf_states)
    
    # Visualization
    # Set the figure size
    plt.figure(figsize=(10, 7))

    # Plotting the actual, measured, and KF paths
    plt.plot(x_true, y_true, 'b-', label="Actual Path", linewidth=2) 
    plt.plot(ekf_states[:, 0], ekf_states[:, 1], 'g--', label="EKF Path", linewidth=2)  
    plt.scatter(x_measured, y_measured, color='r', s=30, label="Sensor Data", alpha=0.5)  

    # Add arrows to show orientation at selected points
    arrow_skip = 10  # Number of points to skip between arrows
    for i in range(0, len(theta_true), arrow_skip):
        plt.arrow(x_true[i], y_true[i], 
                  0.1 * np.cos(theta_true[i]), 0.1 * np.sin(theta_true[i]), 
                  head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        
    # Add arrows to show orientation for KF path
    for i in range(0, ekf_states.shape[0], arrow_skip):
        plt.arrow(ekf_states[i, 0], ekf_states[i, 1], 
                  0.1 * np.cos(ekf_states[i, 2]), 0.1 * np.sin(ekf_states[i, 2]), 
                  head_width=0.05, head_length=0.1, fc='green', ec='green')

    # Marking start and end points for each path
    plt.scatter(x_true[0], y_true[0], color='b', marker='o', s=100, label="Start (Actual)", edgecolor='black')
    plt.scatter(x_true[-1], y_true[-1], color='b', marker='X', s=100, label="End (Actual)", edgecolor='black')
    plt.scatter(ekf_states[0, 0], ekf_states[0, 1], color='g', marker='o', s=100, label="Start (EKF)", edgecolor='black')
    plt.scatter(ekf_states[-1, 0], ekf_states[-1, 1], color='g', marker='X', s=100, label="End (EKF)", edgecolor='black')

    # Adding labels, title, grid, and legend
    plt.xlabel("X Position") 
    plt.ylabel("Y Position") 
    plt.title("Extend Kalman Filter Path Tracking") 
    plt.legend() 
    plt.grid(True) 
    plt.show()

if __name__ == '__main__':
    main()
