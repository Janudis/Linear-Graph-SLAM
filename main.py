import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from functions import *

# world parameters
num_landmarks      = 5        # number of landmarks
N                  = 20       # time steps
world_size         = 100.0    # size of world (square)

# robot parameters
measurement_range  = 50.0     # range at which we can sense landmarks
motion_noise       = 2.0      # noise in robot motion
measurement_noise  = 2.0      # noise in the measurements
distance           = 20.0     # distance by which robot (intends to) move each iteratation


# make_data instantiates a robot, AND generates random landmarks for a given world size and number of landmarks
data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)

# # print out some stats about the data
# time_step = 0
# print('Example measurements: \n', data[time_step][0])
# print('\n')
# print('Example motion: \n', data[time_step][1])

# define a small N and world_size (small for ease of visualization)
N_test = 5
num_landmarks_test = 2
small_world = 10

# # initialize the constraints
# initial_omega, initial_xi = initialize_constraints(N_test, num_landmarks_test, small_world)
# # define figure size
# plt.rcParams["figure.figsize"] = (10,7)
#
# # display omega
# sns.heatmap(DataFrame(initial_omega), cmap='Blues', annot=True, linewidths=.5)
#
# # define  figure size
# plt.rcParams["figure.figsize"] = (1,7)
#
# # display xi
# sns.heatmap(DataFrame(initial_xi), cmap='Oranges', annot=True, linewidths=.5)

# call your implementation of slam, passing in the necessary parameters
mu = slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise)
#print out the resulting landmarks and poses
if(mu is not None):
    # get the lists of poses and landmarks
    # and print them out
    poses, landmarks = get_poses_landmarks(mu,num_landmarks, N)
    print_all(poses, landmarks)