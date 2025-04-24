import geopandas as gpd
from shapely.geometry import Polygon
from geopy.distance import geodesic
import numpy as np
import json
import pickle
import time

class Grid:
    def __init__(self, top_left=(40.0915, 116.1073), bottom_right=(39.7283, 116.7325), granularity=1000):
        # top_left and bottom_right are tuples of (lat, lon)
        self.tl = top_left
        self.br = bottom_right
        # granularity in meters 
        self.g = granularity 

    def create_grid(self, initial_value=None):
        # Calculate the number of rows and columns
        num_rows = int(geodesic(self.tl, (self.br[0], self.tl[1])).meters / self.g)
        num_cols = int(geodesic(self.tl, (self.tl[0], self.br[1])).meters / self.g)

        # Generate grid polygons
        grid_polygons = []
        for row in range(num_rows):
            for col in range(num_cols):
                top_left_lat = self.tl[0] - (row * (self.g / 111111))  # Approx. degrees per meter for latitude
                top_left_lon = self.tl[1] + (col * (self.g / (111111 * np.cos(np.radians(top_left_lat)))))
                bottom_right_lat = top_left_lat - (self.g / 111111)
                bottom_right_lon = top_left_lon + (self.g / (111111 * np.cos(np.radians(bottom_right_lat))))

                polygon = Polygon([(top_left_lon, top_left_lat),
                                (bottom_right_lon, top_left_lat),
                                (bottom_right_lon, bottom_right_lat),
                                (top_left_lon, bottom_right_lat)])

                grid_polygons.append(polygon)

        # Create a GeoDataFrame from the grid polygons
        self.grid = gpd.GeoDataFrame({'geometry': grid_polygons})
        self.grid['label'] = self.grid.index.map(lambda x: [x])

        # Create a dictionary to store the neighbors of each cell
        self.find_neighbor()

    def create_syntetic_grid(self):
        starttime = time.time()
        self.syntetic = True
        # create a syntetic grid considering a cartesian plan from self.tl to self.br considering the granularity as the distance between the points
        # Calculate the number of rows and columns
        num_rows = int(np.ceil((self.br[1] - self.tl[1]) / self.g))
        num_cols = int(np.ceil((self.br[0] - self.tl[0]) / self.g))

        # Generate grid polygons
        grid_polygons = []
        for i in range(num_rows):
            for j in range(num_cols):
                x0 = self.tl[0] + j * self.g
                y0 = self.tl[1] + i * self.g
                x1 = x0 + self.g
                y1 = y0
                x2 = x0 + self.g
                y2 = y0 + self.g
                x3 = x0
                y3 = y0 + self.g
            
                polygon = Polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)])
                grid_polygons.append(polygon)

        self.grid = gpd.GeoDataFrame({'geometry': grid_polygons})
        self.grid['label'] = self.grid.index.map(lambda x: [x])
        endtime = time.time()
       
    def convert_grid_list(self,grid_list):
       
        grid_polygons = []
        for point in grid_list:
            polygon = Polygon([(point[0][0], point[0][1]), (point[0][2], point[0][1]), (point[0][2], point[0][3]), (point[0][0], point[0][3])])
            grid_polygons.append(polygon)

        self.grid = gpd.GeoDataFrame({'geometry': grid_polygons})
        self.grid['label'] = self.grid.index.map(lambda x: [x])

    def generate_grid(self, polygon_list):
        self.grid = gpd.GeoDataFrame({'geometry': polygon_list})
        self.grid['label'] = self.grid.index.map(lambda x: [x])
 

    def save_grid(self, file_name):
        # Save the GeoDataFrame to a file or visualize it as needed
        self.grid.to_file(file_name, driver='GeoJSON')

    def save_grid_pkl(self,file_name):
        with open(file_name, 'wb') as output:
            pickle.dump(self.grid, output)
        
    def load_grid_pkl(self,file_name):
        with open(file_name, 'rb') as input:
            self.grid = pickle.load(input)
        return self.grid
    
    def load_grid(self, file_name):
        self.grid = gpd.read_file(file_name)
        return self.grid

    def get_grid(self):
        return self.grid
    
    def get_grid_centroids(self):
        return self.grid["geometry"].centroid
    
    def find_neighbor(self):
        # Create a dictionary to store the neighbors of each cell
        neighbors = {}

        # Calculate the number of rows and columns
        num_rows = int(geodesic(self.tl, (self.br[0], self.tl[1])).meters / self.g)
        num_cols = int(geodesic(self.tl, (self.tl[0], self.br[1])).meters / self.g)


        for i, row in self.grid.iterrows():
            # Identify the row and column of the current cell
            current_row = i // num_cols
            current_col = i % num_cols

            # Define a list to store the indices of the neighbors
            neighbor_indices = []

            # Check and add the neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the current cell

                    neighbor_row = current_row + dr
                    neighbor_col = current_col + dc

                    # Check if the neighbor is within the grid
                    if 0 <= neighbor_row < num_rows and 0 <= neighbor_col < num_cols:
                        neighbor_index = neighbor_row * num_cols + neighbor_col
                        neighbor_indices.append(neighbor_index)

            neighbors[i] = neighbor_indices

        self.grid['neighbor'] = neighbors.values()