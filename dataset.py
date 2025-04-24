#THIS function is used to generate, and load the sintetic datasets used in the experiments


import numpy as np
import pandas as pd
import pickle
from pathlib import Path


def generate_point_with_direction(pre_point, speed, x_min, x_max, y_min, y_max, target_quadrant, ti=60):
    # Calculate the maximum distance based on speed and time
    distance = (speed * 1000) / 3600 * ti
    distance = np.random.uniform(0, distance)

    angle = 0

    # Determine target angle based on the quadrant
    if target_quadrant == 1:
        angle = np.random.uniform(np.pi / 2, np.pi)  # Quadrant 1
    elif target_quadrant == 2:
        angle = np.random.uniform(0, np.pi / 2)  # Quadrant 2
    elif target_quadrant == 3:
        angle = np.random.uniform(np.pi, 3 * np.pi / 2)  # Quadrant 3
    elif target_quadrant == 4:
        angle = np.random.uniform(3 * np.pi / 2, 2 * np.pi)  # Quadrant 4

    # Calculate the new point
    x = pre_point[0] + distance * np.cos(angle)
    y = pre_point[1] + distance * np.sin(angle)

    # Ensure the point stays within the boundaries
    x = max(min(x, x_max - 0.1), x_min + 0.1)
    y = max(min(y, y_max - 0.1), y_min + 0.1)

    return (x, y)


def generate_points_with_quadrant_movements(num_points, speed, x_min, x_max, y_min, y_max, initial_quadrant,target_vector, ti=60):
    # Calculate the center for the initial quadrant
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    
    distribution_window = 0.2 * num_points

    if initial_quadrant == 1:  # Superior direito
        x_min_point = x_min
        x_max_point = x_center
        y_min_point = y_center
        y_max_point = y_max
    elif initial_quadrant == 2:  # Superior esquerdo
        x_min_point = x_center
        x_max_point = x_max
        y_min_point = y_center
        y_max_point = y_max
    elif initial_quadrant == 3:  # Inferior esquerdo
        x_min_point = x_min
        x_max_point = x_center
        y_min_point = y_min
        y_max_point = y_center
    elif initial_quadrant == 4:  # Inferior direito
        x_min_point = x_center
        x_max_point = x_max
        y_min_point = y_min
        y_max_point = y_center

    point = [(np.random.uniform(x_min_point, x_max_point), np.random.uniform(y_min_point, y_max_point))]

    # Generate the remaining points
        
    for i in range(0, num_points-1):
        point.append(generate_point_with_direction(point[-1], speed, x_min, x_max, y_min, y_max,ti,target_vector[i]))
    
    return point

        

def generate_users_points_with_quadrants(n_users, num_points, speed, x_min, x_max, y_min, y_max, ti=60):
    users_points = []

    # Choose initial quadrant probabilistically
    # initial_quadrant = np.random.randint(1, 5)

    #start uniform

    target_vector = []
    #set_target_quadrante at every 20 points from the number of points
    for i in range(num_points):
        if i % 20 == 0:
            t = np.random.choice([1, 2, 3, 4], p=[0.25, 0.25, 0.25, 0.25])
        target_vector.append(t)

    print(len(target_vector))

    for _ in range(n_users):
        # Choose initial quadrant probabilistically
        initial_quadrant = np.random.randint(1, 5)
        # Choose initial quadrant probabilistically
        users_points.append(generate_points_with_quadrant_movements(num_points, speed, x_min, x_max, y_min, y_max, initial_quadrant,target_vector, ti))
    return users_points

def get_data_boundaries(data):
    num_points = len(data[0])
    n_users = len(data)
    x_min = min(point[0] for pointlist in data for point in pointlist)
    y_min = min(point[1] for pointlist in data for point in pointlist)
    x_max = max(point[0] for pointlist in data for point in pointlist)
    y_max = max(point[1] for pointlist in data for point in pointlist)

    x_min = x_min - 1
    x_max = x_max + 1
    y_min = y_min - 1
    y_max = y_max + 1


    return x_min, y_min, x_max, y_max, num_points, n_users

def generate_point(pre_point, speed, x_min, x_max, y_min, y_max,ti=60):
    # Calculate the distance that the user will travel in the x and y axis considering that x and y are in meters and speed in km/h, and each point is 60 seconds apart
    distance = (speed * 1000) / 3600 * ti
    distance = np.random.uniform(0, distance)

    # Generate a random angle in radians
    angle = np.random.uniform(0, 2 * np.pi)
    # Calculate the new point
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
    
    x = -1
    y = -1
    # Generate the first point
    if distribution == 0:
        points = [(np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max))]
    elif distribution == 1:
           
        std_dev = med/2

        while x < x_min or x > x_max or y <y_min or y > y_max:
            x = np.random.normal(loc=x_center, scale=std_dev)
            y = np.random.normal(loc=y_center, scale=std_dev)

        points = [(x,y)]

    elif distribution == 2:
        x = np.random.exponential(scale=std_dev)
        y = np.random.exponential(scale=std_dev)
        if np.random.uniform(-1,1)<0:
            x = -x
        if np.random.uniform(-1,1)<0:
            y = -y

        points = [(x+x_center,y+y_center)]

    # Generate the remaining points
    for i in range(num_points - 1):
        points.append(generate_point(points[-1], speed, x_min, x_max, y_min, y_max,ti))
    return points



def generate_users_points(n_users, num_points, speed, x_min, x_max, y_min, y_max,distribution=0,ti=60):
    users_points = []
    for i in range(n_users):
        users_points.append(generate_points(num_points, speed, x_min, x_max, y_min, y_max,distribution,ti))
    return users_points


def load_data(data_path):
    x_min = 0
    x_max = 10000
    y_min = 0
    y_max = 10000

    return pd.read_pickle(data_path), x_min, y_min, x_max, y_max

def save_data(data,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(data,f)


def aggregate_distribution_every_20_points_4(data, x_min, x_max, y_min, y_max):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    max_points = max(len(user_points) for user_points in data)
    interval = 20

    for start in range(0, max_points, interval):
        quadrant_counts = {1: 0, 2: 0, 3: 0, 4: 0}

        for user_points in data:
            for point in user_points[start:start + interval]:
                x, y = point
                if x >= x_center and y >= y_center:
                    quadrant_counts[1] += 1
                elif x < x_center and y >= y_center:
                    quadrant_counts[2] += 1
                elif x < x_center and y < y_center:
                    quadrant_counts[3] += 1
                elif x >= x_center and y < y_center:
                    quadrant_counts[4] += 1

        print(f"Intervalo {start + 1} a {start + interval}: {quadrant_counts}")


def aggregate_distribution_every_20_points_100(data, x_min, x_max, y_min, y_max):
    x_divisions = 10
    y_divisions = 10

    # Calculate the width and height of each cell in the grid
    cell_width = (x_max - x_min) / x_divisions
    cell_height = (y_max - y_min) / y_divisions

    
    max_points = max(len(user_points) for user_points in data)
    interval = 20

    for start in range(0, max_points, interval):
        # Initialize a grid to count points in each cell
        grid_counts = np.zeros((x_divisions, y_divisions), dtype=int)

        for user_points in data:
            for point in user_points[start:start + interval]:
                x, y = point

                # Determine the cell in which the point falls
                x_idx = int((x - x_min) // cell_width)
                y_idx = int((y - y_min) // cell_height)

                # Ensure indices stay within bounds
                x_idx = min(x_idx, x_divisions - 1)
                y_idx = min(y_idx, y_divisions - 1)

                # Increment the count for the corresponding cell
                grid_counts[x_idx, y_idx] += 1

        print(f"Intervalo {start + 1} a {start + interval}:")
        print(grid_counts)
        print()