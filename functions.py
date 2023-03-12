from robot import robot
from math import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------
# this helper function displays the world that a robot is in
# it assumes the world is a square grid of some given size
# and that landmarks is a list of landmark positions(an optional argument)
def display_world(world_size, position, landmarks=None):
    # using seaborn, set background grid to gray
    sns.set_style("dark")

    # Plot grid of values
    world_grid = np.zeros((world_size + 1, world_size + 1))

    # Set minor axes in between the labels
    ax = plt.gca()
    cols = world_size + 1
    rows = world_size + 1

    ax.set_xticks([x for x in range(1, cols)], minor=True)
    ax.set_yticks([y for y in range(1, rows)], minor=True)

    # Plot grid on minor axes in gray (width = 1)
    plt.grid(which='minor', ls='-', lw=1, color='white')

    # Plot grid on major axes in larger width
    plt.grid(which='major', ls='-', lw=2, color='white')

    # Create an 'o' character that represents the robot
    # ha = horizontal alignment, va = vertical
    ax.text(position[0], position[1], 'o', ha='center', va='center', color='r', fontsize=30)

    # Draw landmarks if they exists
    if (landmarks is not None):
        # loop through all path indices and draw a dot (unless it's at the car's location)
        for pos in landmarks:
            if (pos != position):
                ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)

    # Display final result
    plt.show()


# --------
# this routine makes the robot data
# the data is a list of measurements and movements: [measurements, [dx, dy]]
# collected over a specified number of time steps, N
#
def make_data(N, num_landmarks, world_size, measurement_range, motion_noise,
              measurement_noise, distance):
    # check that data has been made
    try:
        check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise)
    except ValueError:
        print('Error: You must implement the sense function in robot_class.py.')
        return []

    complete = False

    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    while not complete:

        data = []

        seen = [False for row in range(num_landmarks)]

        # guess an initial motion
        orientation = random.random() * 2.0 * pi
        dx = cos(orientation) * distance
        dy = sin(orientation) * distance

        for k in range(N - 1):

            # collect sensor measurements in a list, Z
            Z = r.sense()

            # check off all landmarks that were observed
            for i in range(len(Z)):
                seen[Z[i][0]] = True

            # move
            while not r.move(dx, dy):
                # if we'd be leaving the robot world, pick instead a new direction
                orientation = random.random() * 2.0 * pi
                dx = cos(orientation) * distance
                dy = sin(orientation) * distance

            # collect/memorize all sensor and motion data
            data.append([Z, [dx, dy]])

        # we are done when all landmarks were observed; otherwise re-run
        complete = (sum(seen) == num_landmarks)

    print(' ')
    print('Landmarks: ', r.landmarks)
    print(r)

    return data


def check_for_data(num_landmarks, world_size, measurement_range, motion_noise, measurement_noise):
    # make robot and landmarks
    r = robot(world_size, measurement_range, motion_noise, measurement_noise)
    r.make_landmarks(num_landmarks)

    # check that sense has been implemented/data has been made
    test_Z = r.sense()
    if (test_Z is None):
        raise ValueError


def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''

    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    rows, cols = 2 * (N + num_landmarks), 2 * (N + num_landmarks)  # double the size for x and y
    ## TODO: Define the constraint matrix, Omega, with two initial "strength" values
    ## for the initial x, y location of our robot
    omega = np.zeros((rows, cols))
    omega[0][0] = 1
    omega[1][1] = 1

    ## TODO: Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    xi = np.zeros((rows, 1))
    xi[0][0] = world_size / 2
    xi[1][0] = world_size / 2

    return omega, xi


def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    measurement_corr = 1.0 / measurement_noise
    motion_corr = 1.0 / motion_noise

    ## TODO: Use your initilization to create constraint matrices, omega and xi
    omega, xi = initialize_constraints(N, num_landmarks, world_size)
    ## TODO: Iterate through each time step in the data
    ## get all the motion and measurement data as you iterate
    for timestep in range(len(data)):
        measurements = data[timestep][0]
        motion = data[timestep][1]
        t = timestep * 2

        ## TODO: update the constraint matrix/vector to account for all *measurements*
        ## this should be a series of additions that take into account the measurement noise
        for measurement in range(len(measurements)):
            m = 2 * (N + measurements[measurement][0])
            # x = measurement[1]
            # y = measurement[2]

            for xy in range(2):  # 0 for x, and 1 for y
                omega[t + xy][t + xy] += 1.0 / measurement_noise  # fill the diagonals with +1
                omega[m + xy][m + xy] += 1.0 / measurement_noise  # fill the diagonals with +1
                omega[t + xy][m + xy] += -1.0 / measurement_noise  # fill the off-diagonals with -1
                omega[m + xy][t + xy] += -1.0 / measurement_noise  # fill the off-diagonals with -1

                xi[t + xy][0] += -measurements[measurement][1 + xy] / measurement_noise  # update the measurement
                xi[m + xy][0] += measurements[measurement][1 + xy] / measurement_noise

        ## TODO: update the constraint matrix/vector to account for all *motion* and motion noise
        for xy in range(2):
            omega[t + xy][t + xy + 2] += -1.0 / motion_noise  # fill the off-diagonals with -1
            omega[t + xy + 2][t + xy] += -1.0 / motion_noise  # fill the off-diagonals with -1

            omega[t + xy][t + xy] += 1.0 / motion_noise  # fill the diagonals with +1
            omega[t + xy + 2][t + xy + 2] += 1.0 / motion_noise  # fill the diagonals with +1

            xi[t + xy][0] += -motion[xy] / motion_noise  # update the motion
            xi[t + xy + 2][0] += motion[xy] / motion_noise

            # xi[t][0]      += -motion[0] /motion_noise
        # xi[t + 1][0]   += -motion[1] /motion_noise
        # xi[t + 2][0]   += motion[0] /motion_noise
        # xi[t + 3][0]   += motion[1] /motion_noise
    ## TODO: After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    mu = np.matrix(omega).I * xi
    return mu  # return `mu`

def get_poses_landmarks(mu, num_landmarks, N):
    # create a list of poses
    poses = []
    for i in range(N):
        poses.append((mu[2*i].item(), mu[2*i+1].item()))

    # create a list of landmarks
    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2*(N+i)].item(), mu[2*(N+i)+1].item()))

    # return completed lists
    return poses, landmarks

def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('['+', '.join('%.3f'%p for p in poses[i])+']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('['+', '.join('%.3f'%l for l in landmarks[i])+']')