# This file we will generate points for each user in the cartesian plane to simulate the data that we will use in the experiment4.py

import numpy as np

def generate_point(pre_point, speed, x_min, x_max, y_min, y_max,ti=60):
    # Calculate the distance that the user will travel in the x and y axis considering that x and y are in meters and speed in km/h, and each point is 60 seconds apart
    distance = (speed * 1000) / 3600 * ti
    distance = np.random.uniform(0, distance)

    angle = np.random.uniform(0, 2 * np.pi)

    x = pre_point[0] + distance * np.cos(angle)
    y = pre_point[1] + distance * np.sin(angle)

    # Check if the new point is within the limits
    if x < x_min:
        x = x_min + 0.1
    if x > x_max:
        x = x_max - 0.1
    if y < y_min:
        y = y_min + 0.1
    if y > y_max:
        y = y_max - 0.1

    return (x, y)

def generate_points(num_points, speed, x_min, x_max, y_min, y_max,distribution,ti):
    
    x_center = (x_max - x_min) / 2
    y_center = (y_max - y_min) / 2

    med = (x_max - x_min) / 2
    
    # Generate the first point
    if distribution == 0:
        point = [(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))]
    elif distribution == 1:
           
        std_dev = med

        x = np.random.normal(loc=x_center, scale=std_dev)
        y = np.random.normal(loc=y_center, scale=std_dev)

        point = [(x,y)]

    elif distribution == 2:
        x = np.random.exponential(scale=std_dev)
        y = np.random.exponential(scale=std_dev)
        if np.random.uniform(-1,1)<0:
            x = -x
        if np.random.uniform(-1,1)<0:
            y = -y

        point = [(x+x_center,y+y_center)]

    # Generate the remaining points
    for i in range(num_points - 1):
        point.append(generate_point(point[-1], speed, x_min, x_max, y_min, y_max,ti))
    return point

# calculate the distance btw two points in cartesian plane
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def generate_users_points(n_users, num_points, speed, x_min, x_max, y_min, y_max,distribution=0,ti=60):
    users_points = []
    for i in range(n_users):
        users_points.append(generate_points(num_points, speed, x_min, x_max, y_min, y_max,distribution,ti))
    return users_points
