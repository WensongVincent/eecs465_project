import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, motion_noise, sensor_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.motion_noise = motion_noise
        self.sensor_noise = sensor_noise
        
    def predict(self, control_input, dt):
        v_l, v_r = control_input
        x, y, theta = self.state
        L = 0.5  # 假设轮子间距为0.5米

        # 运动模型
        delta_d = (v_l + v_r) / 2.0 * dt
        delta_theta = (v_r - v_l) / L * dt
        x += delta_d * np.cos(theta)
        y += delta_d * np.sin(theta)
        theta += delta_theta

        # 更新状态
        self.state = np.array([x, y, theta])

        # 更新协方差
        F = np.array([[1, 0, -delta_d * np.sin(theta)],
                      [0, 1, delta_d * np.cos(theta)],
                      [0, 0, 1]])

        Q = np.diag([self.motion_noise, self.motion_noise, self.motion_noise * dt])
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, measurements):
        for i, landmark in enumerate(self.landmarks):
            distance, angle = measurements[i]
            x, y, theta = self.state

            # 计算到地标的预期距离和角度
            dx = landmark[0] - x
            dy = landmark[1] - y
            d = np.sqrt(dx**2 + dy**2)
            expected_angle = np.arctan2(dy, dx) - theta

            # 观测模型的雅可比矩阵
            H = np.array([[-dx / d, -dy / d, 0],
                          [dy / (d**2), -dx / (d**2), -1]])

            # 计算卡尔曼增益
            S = H @ self.covariance @ H.T + self.sensor_noise
            K = self.covariance @ H.T @ np.linalg.inv(S)

            # 更新状态
            z = np.array([distance, angle])
            expected_z = np.array([d, expected_angle])
            self.state = self.state + K @ (z - expected_z)

            # 更新协方差
            self.covariance = (np.eye(3) - K @ H) @ self.covariance

    def get_state(self):
        return self.state

# Read data from file
def read_path_from_file(file_path, T_MAX):
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

    return np.array(path)

# Simulate lidar measurements 
def simulate_lidar_measurements():
    

def main():
    ################ read path from recorded data ################
    path = read_path_from_file('path_maze.txt')
    x_path, y_path, theta_path = path[:,0], path[:,1], path[:,2]

    ################ Extend Kalman Filter ################
    # EKF initialization
    initial_state = path[0,:]  # x0: x, y, θ
    initial_covariance = np.eye(3)  # p0: initial covarience
    motion_noise = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])  # random motion noise for test
    sensor_noise = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])  # random sensor noise for test
    
    ekf = ExtendedKalmanFilter(initial_state, initial_covariance, motion_noise, sensor_noise)

    # loop through all the path to estimate the state of the pr2 robot
    # 用于存储EKF状态的列表
    ekf_states = []
    # 遍历路径
    for i in range(1, path.shape[1]):
        # 假设控制输入是基于位置差的
        control_input = [x_path[i] - x_path[i-1], y_path[i] - y_path[i-1]]
        ekf.predict(control_input, dt = 1.0)  # 预测步骤

        # 模拟从激光雷达得到的测量数据
        # 此处为了简化，我们使用真实位置加上一些随机噪声
        measurements = [(np.sqrt((lm[0] - x_path[i])**2 + (lm[1] - y_path[i])**2) + np.random.normal(0, 0.5),
                         np.arctan2(lm[1] - y_path[i], lm[0] - x_path[i]) - theta_path[i] + np.random.normal(0, 0.1)) for lm in landmarks]
        ekf.update(measurements)  # 更新步骤

        ekf_states.append(ekf.get_state())

    ekf_states = np.array(ekf_states)
    
    # visualization
    plt.figure(1)
    plt.plot(x_path, y_path, label = "Actual Path")
    plt.scatter(ekf_states[:, 0], ekf_states[:, 1], label = "EKF Path")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
