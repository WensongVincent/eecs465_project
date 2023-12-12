import numpy as np

################ linear model ################
# A matrix - state matrix
A = np.eye(3)

# B matrix - control matrix
B = np.eye(3)

# C matrix - observation matrix
C = np.eye(3)

################ non-linear model ################







################ Sensor parameter ################
# R matrix - motion noise covarience matrix
R = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

# Q matrix - sensor noise covarience matrix
Q = np.array([[0.1, 0.01, 0.01], [0.01, 0.1, 0.01], [0.01, 0.01, 0.1]])