########################################################################################################
#
#   Experiment 10: Consuming budget ate every timestamp 
#   Base Line: Uniform Grid                                                                             #
#   Two Round Execution:                                                                                #
#   AP1(NOT WORKING): - Adaptative Grip with 1 split at every timestamp, using a fixed size grid as base.            #
#        - The split has two executions rounds, the first one with a budget of bp and the               # 
#        - second one with a budget of 1-bp where the report will be collected based on the new grid.   # 
#   AP2(ALOG-2R): - Adaptative Grip with 1 splits after x timestamp, using a the previous grid as base           #
#        - The split has two executions rounds, the first one with a budget of bp and the               #
#        - second one with a budget of 1-bp where the report will be collected based on the new grid.   #
#   One Round Execution:                                                                                #
#   AP3(ALOG-1Ra): - Adaptative Grip with 1 splits after x timestamp, using the previous grid as base             #
#        - The split has one execution only. THe new grid will be used only in the next timestamp.      #
#   AP4(ALOG-1Rb: - Adaptative Grip with 1 splits after x timestamp, using a fixed size grid as base             #
#        - The split has one execution only. THe new grid will be used only in the next timestamp.
# 
#   ALL The Approaches will be executed with the following protocols: LOSUE
#                                    #
#########################################################################################################

from generate_sintetic_data import generate_users_points
import multiprocessing
import csv
## Structure imports

from grid import Grid
from grid2 import Grid2
from shapely.geometry import Point
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from multiprocessing import Pool
import os
from functools import partial
import math
from shapely.ops import unary_union
import copy
import pickle
import threading
import sys

from dataset import load_data

import warnings; warnings.filterwarnings('ignore')

from sys import maxsize
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

from FO.LDP.protocols import setting_seed
from FO.LDP.protocols import LOLOHA_Client, LOLOHA_Aggregator
from FO.LDP.protocols import RAPPOR_Client, RAPPOR_Client_TAU, RAPPOR_Aggregator # [1]
from FO.LDP.protocols import L_GRR_Client, L_GRR_Client_TAU, L_GRR_Aggregator 
from FO.LDP.protocols import L_OSUE_Client, L_OSUE_Client_TAU, L_OSUE_Aggregator
from FO.LDP.protocols import LOLOHA_Client_TAU, LOLOHA_Aggregator_TAU 

#####################################################################
#                           Auxiliar Functions                      #
#####################################################################

def encode_dada(data,grid):
    enc_data = []

    for i in range(len(data)):
        enc_locations = []
        for tau in range(len(data[i])):
            P = Point(data[i][tau][0],data[i][tau][1])
        
            id = -1
            for j in (range(len(grid))):
                if(grid.iloc[j]['geometry'].contains(P)):
                    id = j
                    break
            if id == -1:
                print("Error: id = -1")
                print("P:",P)
                print("-------")
            enc_locations.append(id)
        enc_data.append(enc_locations)

    return enc_data


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

def get_real_freq(data,k):
    real_freq = np.zeros(k)
    
    for item in data:
        real_freq[item]+=1
    
    real_freq = real_freq / sum(real_freq)

    return real_freq 

def split_grid(grid,est_freq,num_users,fr,check_norm=False,count_check=False):
    

    try:
        if count_check:
            est_freq = [int(f * 10000) for f in est_freq]
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Length of est_freq: {len(est_freq)}")
        for f in est_freq:
            if not isinstance(f, float) or f != f:  # f != f is a way to check for NaN
                print(f"Problematic value: {f}")
        raise  # Re-raise the error to ensure you notice the issue

    grid_list = []
    for i in range(len(grid)):
        grid_list.append([grid.iloc[i]['geometry'].bounds,grid.iloc[i]['label'],est_freq[i]])
    
    new_grid_list = []
    # print("tr:",fr)
    # print("type:",type(fr))
    for g in grid_list:
        if g[2] > fr:
            new_grid_list += naive_split(g,fr)
        else:
            # print("count_grid_cell:",g[2])
            new_grid_list.append(g)

            
    x_min,y_min,x_max,y_max = 0,0,0,0
    count = 0

    new_grid_instance = Grid((x_min,y_min),(x_max,y_max),count)
    new_grid_instance.convert_grid_list(new_grid_list)

    return new_grid_instance.grid

def naive_split_fr(cell,fr):
    
    #divide a célula em 4 considerando que os dados são uniformes, portanto o count de cada nova cell é igual a cell[2]/4. Nova conta precisa ser inteira e a soma igual a cell[2]
    new_cells = []
    new_fr = cell[2] / 4
    
    xmin = cell[0][0]
    ymin = cell[0][1]
    xmax = cell[0][2]
    ymax = cell[0][3]

    granularity = (xmax - xmin)/2



    for i in range(2):
        for j in range(2):
            x_min = xmin  + j * granularity
            y_min = ymin + i * granularity
            x_max = x_min + granularity
            y_max = y_min + granularity
            
            new_cell = [[x_min,y_min,x_max,y_max],cell[1],new_fr]
                
            if  new_cell[2] > fr:
                new_cells += naive_split_fr(new_cell,fr)
            else:
                new_cells.append(new_cell)   

    
    return new_cells

def naive_split(cell,fr):
    
    #divide a célula em 4 considerando que os dados são uniformes, portanto o count de cada nova cell é igual a cell[2]/4. Nova conta precisa ser inteira e a soma igual a cell[2]
    new_cells = []
    new_fr = cell[2] / 4
    
    xmin = cell[0][0]
    ymin = cell[0][1]
    xmax = cell[0][2]
    ymax = cell[0][3]

    granularity = (xmax - xmin)/2


    for i in range(2):
        for j in range(2):
            x_min = xmin  + j * granularity
            y_min = ymin + i * granularity
            x_max = x_min + granularity
            y_max = y_min + granularity
            
            new_cell = [[x_min,y_min,x_max,y_max],cell[1],new_fr]
                
            if  new_cell[2] > fr:
                new_cells += naive_split_fr(new_cell,fr)
            else:
                new_cells.append(new_cell)   

    
    return new_cells


def gen_execution_data(data_set_type, data_set_distribution,file_name,ti,num_points,n_users,speed):
    # Generate syntetic Data

    speed = speed
    x_min = 0
    x_max = 10000
    y_min = 0
    y_max = 10000
    
    ## Os dados sintéticos são gerados com um timestamp de 60s, então vamos considerar um tempo de modificação do
    ## heatmap de 5 minutos, ou seja, 5 timestamps - Podemos testar outros valores depois
   

    ######################################
    ##   distribution:                   #
    ##   - 0: uniform distribution       #
    ##   - 1: normal distribution        #
    ##   - 2: exponential distribution   #
    ######################################
    
    
    if data_set_distribution == 'u':
        distribution = 0
    elif data_set_distribution == 'n':
        distribution = 1
    elif data_set_distribution == 'e':
        distribution = 2

    if data_set_type == "s":
        data = generate_users_points(n_users, num_points, speed, x_min, x_max, y_min, y_max, distribution, ti)
        n_users = len(data)

    else :
        if file_name != None:
            data = pd.read_pickle(file_name)
            x_min, y_min, x_max, y_max, num_points, n_users = get_data_boundaries(data)
            data = data[:n_users]

        else:
            print("File Name not set!")
            return

    return data, x_min, y_min, x_max, y_max
    # return data

#####################################################################
#                             APROACHES                             #   
#####################################################################
############################### AP1 #################################
#####################################################################

def AP1(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False):
    
    #AP1 has grid adaptation every ts. So, we need to create a new memo_vector for each user at each ts. So 
    # the same user is always using a different memo vector for each ts. So we dont need memoization
    memoization = False

    #ti = 60 #seconds
    execution_times = 2

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points

    eps_perm_base = eps_perm

    alpha = 1/tau
        
    # eps_perm_per_tau = eps_perm_base / num_points
    
    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()

    num_users = len(data)

    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    grid_coordenates = []
    tr_vector = []
    est_frequency_vector = []
    
    # client - side
    
    k =  len(grid_base)

    grid_size_base = k

    final_budget_users = [0 for _ in range(n_users)]


    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        coord_vector = []
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)

        k = len(grid)

        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,grid)   

        
        #### AP1 ONLY ####        
        eps_perm1 = eps_perm * bp
        eps_perm2 = eps_perm * (1-bp)
        eps_11 = eps_perm1 * alpha
        eps_12 = eps_perm2 * alpha
        ##################

        
        for e in range(execution_times):

            reports = []

              
            if e != 0:
               
                if tr_count:
                    tr_value = int(n_users * tr)
                    #tr_value = int(n_users/k) # looking for uniformity
                else:
                    tr_value = 1 / k

                # if any values from est_freq is different from 0, we will split the grid
                if any(est_freq):
                    grid = split_grid(grid, est_freq,num_users, tr_value, check_norm,tr_count)

                    k = len(grid)
                    data_encoded = encode_dada(data_tau,grid)

                eps_perm = eps_perm2
                eps_1 = eps_12
                
            else:
                eps_perm = eps_perm1
                eps_1 = eps_11


            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
                                            - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
                                                + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
                                                + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
                                                    / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]  

            for i in range(n_users):
                if protocol == 'RAPPOR':
                    report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
                if protocol == 'LGRR':
                    report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
                if protocol == 'LOSUE':
                    report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
                if protocol == 'LOLOHA':
                    report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)


                reports.append(report)
                final_budget_users[i] += budget_used


            ts_values = []
            for loc in data_encoded:
                ts_values.append(loc[0])
        
            # Server-Side

            real_freq = get_real_freq(ts_values,k)
            if protocol == 'RAPPOR':
                est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm, eps_1)
            if protocol == 'LGRR':
                est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm, eps_1)
            if protocol == 'LOSUE':
                est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm, eps_1)
            if protocol == 'LOLOHA':
                est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm, eps_1, g)
        
              
            est_fr_vector = [fr for fr in est_freq]

        count_vector = [0 for _ in range(k)]

        for i in data_encoded:
            count_vector[i[0]] += 1

        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        # grid_coordenates.append((coord_vector,count_vector))   
        est_frequency_vector.append(est_fr_vector)     
        
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        lst_mse.append(mean_squared_error(real_freq, est_freq))
        lst_mae.append(mean_absolute_error(real_freq, est_freq))
    
    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points))
    

    return results, grid_coordenates

#####################################################################
############################### AP2 #################################
#####################################################################

def AP2(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):

    # ti = 60 #seconds
    execution_times = 2
    # jump = 5 * 60 / ti #(every 5 minutes we will change the base grid)

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points

    eps_perm_base = eps_perm

    alpha = 1/tau

    num_users = len(data)
        
    # eps_perm_base = eps_perm

    # eps_perm_per_tau = eps_perm_base / num_points
    
    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()

    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    grid_coordenates = []
    tr_vector = []
    est_frequency_vector = []
    
    # client - side
    
    k =  len(grid_base)

    grid_size_base = k

    if protocol == 'LOLOHA':
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
                                      - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
                                        + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
                                          + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
                                            / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]


    final_budget_users = [0 for _ in range(n_users)]


    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        coord_vector = []
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)

        k = len(grid)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,grid)   

        
        #### 0, 5,10,15,20
        if t % w == 0:
            execution_times = 2
            eps_perm1 = eps_perm * bp
            eps_perm2 = eps_perm * (1-bp)
            eps_11 = eps_perm1 * alpha
            eps_12 = eps_perm2 * alpha
        else:   
            execution_times = 1
            eps_perm1 = eps_perm
            eps_perm2 = eps_perm
            eps_11 = eps_perm1 * alpha
            eps_12 = eps_perm2 * alpha
    
        for e in range(execution_times):

            reports = []
            
            if e != 0:
                memoization = False
               
                if tr_count:
                    # tr_value = int(n_users * tr)
                    tr_value = int(10000 * tr)
                    #tr_value = int(n_users/k) # looking for uniformity
                else:
                    tr_value = 1 / k
                grid_base = split_grid(grid, est_freq, num_users, tr_value, check_norm,tr_count)

                k = len(grid_base)
                data_encoded = encode_dada(data_tau,grid)

                eps_perm_temp = eps_perm2
                eps_1_temp = eps_12

                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]
                
                # memoization = False

            else:
                memoization = True
                eps_perm_temp = eps_perm1
                eps_1_temp = eps_11

                # memoization = True


            for i in range(n_users):
                if protocol == 'RAPPOR':
                    report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LGRR':
                    report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOSUE':
                    report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOLOHA':
                    report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)


                reports.append(report)
                final_budget_users[i] += budget_used


            ts_values = []
            for loc in data_encoded:
                ts_values.append(loc[0])
        
            # Server-Side

            real_freq = get_real_freq(ts_values,k)
            if protocol == 'RAPPOR':
                est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm_temp, eps_1_temp,notmsub)
            if protocol == 'LGRR':
                est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm_temp, eps_1_temp,notmsub)
            if protocol == 'LOSUE':
                est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm_temp, eps_1_temp,notmsub)
            if protocol == 'LOLOHA':
                est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm_temp, eps_1_temp, g,notmsub)
        
            est_fr_vector = [fr for fr in est_freq]

        count_vector = [0 for _ in range(k)]

        for i in data_encoded:
            count_vector[i[0]] += 1

        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        # grid_coordenates.append((coord_vector,count_vector))   
        est_frequency_vector.append(est_fr_vector)     
        
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        est_freq = np.nan_to_num(est_freq)
        real_freq = np.nan_to_num(real_freq)
        
        lst_mse.append(mean_squared_error(real_freq, est_freq))
        lst_mae.append(mean_absolute_error(real_freq, est_freq))
    
    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points))
    

    return results, grid_coordenates


#####################################################################
############################### AP3 #################################
#####################################################################

def AP3(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):

    # ti = 60 #seconds
    # execution_times = 2
    # jump = 5 * 60 / ti #(every 5 minutes we will change the base grid)

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    alpha = 1/tau

    num_users = len(data)

    # eps_perm_per_tau = eps_perm_base / num_points
    
    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()

    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    grid_coordenates = []
    tr_vector = []
    est_frequency_vector = []
    
    # client - side
    
    k =  len(grid_base)

    grid_size_base = k

    if protocol == 'LOLOHA':
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]  

    final_budget_users = [0 for _ in range(n_users)]


    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        coord_vector = []
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)

        k = len(grid)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,grid)   

        
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        eps_1 = eps_perm * alpha

        memoization = True

        #### 0, 5,10,15,20
        if t % w == 0:
            grid_reconstrution = True
        else:   
            grid_reconstrution = False

        reports = []
            
           

        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)


            reports.append(report)
            final_budget_users[i] += budget_used


        ts_values = []
        for loc in data_encoded:
            ts_values.append(loc[0])
    
        # Server-Side

        real_freq = get_real_freq(ts_values,k)
        if protocol == 'RAPPOR':
            est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LGRR':
            est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm, eps_1,notmsub)
        if protocol == 'LOSUE':
            est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LOLOHA':
            est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm, eps_1, g,notmsub)
    
        est_fr_vector = [fr for fr in est_freq]

        count_vector = [0 for _ in range(k)]

        for i in data_encoded:
            count_vector[i[0]] += 1

        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        # grid_coordenates.append((coord_vector,count_vector))   
        est_frequency_vector.append(est_fr_vector)     
            
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        lst_mse.append(mean_squared_error(real_freq, est_freq))
        lst_mae.append(mean_absolute_error(real_freq, est_freq))

        ### Reconstrution ###
        if grid_reconstrution:
            if tr_count:
                # tr_value = int(n_users * tr)
                tr_value = int(10000 * tr)
                #tr_value = int(n_users/k) # looking for uniformity
            else:
                tr_value = 1 / k
            grid_base = split_grid(grid, est_freq, num_users, tr_value, check_norm,tr_count)

            k = len(grid_base)
            data_encoded = encode_dada(data_tau,grid)

            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]

    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points))
    

    return results, grid_coordenates


def AP4(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):

    # ti = 60 #seconds
    # jump = 5 * 60 / ti #(every 5 minutes we will change the base grid)

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    alpha = 1/tau   

    num_users = len(data) 
    
    # eps_perm_per_tau = eps_perm_base / num_points
    
    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()

    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    grid_coordenates = []
    tr_vector = []
    est_frequency_vector = []
    
    # client - side
    
    k =  len(grid_base)

    grid_size_base = k

    if protocol == 'LOLOHA':
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
                                      - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
                                        + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
                                          + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
                                            / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]   

    final_budget_users = [0 for _ in range(n_users)]

    
    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        coord_vector = []
        tr_vector = []
        
        #### 0, 5,10,15,20
        if t % w == 0:
            grid_reconstrution = True
            grid = copy.deepcopy(grid_base)
        else:   
            grid_reconstrution = False


        k = len(grid)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,grid)   

        
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        eps_1 = eps_perm * alpha

        memoization = True

        

        reports = []
            
    
        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)


            reports.append(report)
            final_budget_users[i] += budget_used


        ts_values = []
        for loc in data_encoded:
            ts_values.append(loc[0])
    
        # Server-Side

        real_freq = get_real_freq(ts_values,k)
        if protocol == 'RAPPOR':
            est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LGRR':
            est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm, eps_1,notmsub)
        if protocol == 'LOSUE':
            est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LOLOHA':
            est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm, eps_1, g,notmsub)
    
        est_fr_vector = [fr for fr in est_freq]

        count_vector = [0 for _ in range(k)]

        for i in data_encoded:
            count_vector[i[0]] += 1

        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        # grid_coordenates.append((coord_vector,count_vector))   
        est_frequency_vector.append(est_fr_vector)     
            
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        lst_mse.append(mean_squared_error(real_freq, est_freq))
        lst_mae.append(mean_absolute_error(real_freq, est_freq))

        ### Reconstrution ###
        if grid_reconstrution:
            if tr_count:
                # tr_value = int(n_users * tr)
                tr_value = int(10000 * tr)
                #print("tr_value:",tr_value)
                #tr_value = int(n_users/k) # looking for uniformity
            else:
                tr_value = 1 / k
            grid = split_grid(grid, est_freq, num_users, tr_value, check_norm,tr_count)

            k = len(grid)
            data_encoded = encode_dada(data_tau,grid)

            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]

            
    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points))
    

    return results, grid_coordenates



#####################################################################
############################ Uniform ################################
#####################################################################

def Uniform(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    
    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    alpha = 1/tau

    # eps_perm_per_tau = eps_perm_base / num_points
    
    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()

    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    grid_coordenates = []
    tr_vector = []
    est_frequency_vector = []
    
    # client - side
    
    k =  len(grid_base)

    grid_size_base = k

    if protocol == 'LOLOHA':
        print("eps_lo:",eps_perm)
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]    

    final_budget_users = [0 for _ in range(n_users)]


    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        coord_vector = []
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)

        k = len(grid)

        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,grid)   

        
        reports = []
        
    
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        eps_1 = alpha * eps_perm

        memoization = True

        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)

            reports.append(report)
            final_budget_users[i] += budget_used


        ts_values = []
        for loc in data_encoded:
            ts_values.append(loc[0])
        
        # Server-Side

        real_freq = get_real_freq(ts_values,k)
        if protocol == 'RAPPOR':
            est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LGRR':
            est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm, eps_1,notmsub)
        if protocol == 'LOSUE':
            est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
        if protocol == 'LOLOHA':
            est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm, eps_1, g,notmsub)
        
        est_fr_vector = [fr for fr in est_freq]

        count_vector = [0 for _ in range(k)]

        for i in data_encoded:
            count_vector[i[0]] += 1

        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        grid_coordenates.append((coord_vector,count_vector))   
        est_frequency_vector.append(est_fr_vector)     
        
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        lst_mse.append(mean_squared_error(real_freq, est_freq))
        lst_mae.append(mean_absolute_error(real_freq, est_freq))
    
    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points))
    

    return results, grid_coordenates
    

#####################################################################
############################# L-GRR #################################
#####################################################################

def grr_execution(results,alpha,eps_perm,bp,w,cell_size,structure,seed,tr,check_norm,tr_count,data, data_info):
    

    for s in structure:
        if s == "Uniform":
            results, grid_coordenates = Uniform("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count) 
        if s == "AP1":
            results, grid_coordenates = AP1("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count)
        if s == "AP2":
            results, grid_coordenates = AP2("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count)    
        if s == "AP3":
            results, grid_coordenates = AP3("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count)
        if s == "AP4":
            results, grid_coordenates = AP4("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count)
        

    return results, grid_coordenates, data
#########################################################################################
#########################################################################################


## RAPPOR ###############################################################################

def rappor_execution(results,seed,eps_perm,bp,w,cell_size,structure,tr, check_norm, tr_count, data, data_info,num_points,notmsub):
    
    
    if structure == "Uniform":
        results, grid_coordenates = Uniform("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub) 
    if structure == "AP1":
        results, grid_coordenates = AP1("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub)
    if structure == "AP2":
        results, grid_coordenates = AP2("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub)  
    if structure == "AP3":
        results, grid_coordenates = AP3("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub)  
    if structure == "AP4":
        results, grid_coordenates = AP4("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub)  


    return results, grid_coordenates, data

#########################################################################################
#########################################################################################


## RAPPOR ###############################################################################

def losue_execution(results,seed,eps_perm,bp,w,cell_size,structure,tr,check_norm,tr_count, data, data_info,num_points,notmsub):
        
    
    if structure == "Uniform":
        results, grid_coordenates = Uniform("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub) 
    if structure == "AP1":
        results, grid_coordenates = AP1("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)
    if structure == "AP2":
        results, grid_coordenates = AP2("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)    
    if structure == "AP3":
        results, grid_coordenates = AP3("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)    
    if structure == "AP4":
        results, grid_coordenates = AP4("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)  


    
    return results, grid_coordenates, data


## LOLOHA ###############################################################################

def loloha_execution(results,seed,eps_perm,bp,w,cell_size,structure,tr,check_norm,tr_count, data, data_info,num_points,notmsub):
        
    
    if structure == "Uniform":
        results, grid_coordenates = Uniform("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub) 
    if structure == "AP1":
        results, grid_coordenates = AP1("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)
    if structure == "AP2":
        results, grid_coordenates = AP2("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)    
    if structure == "AP3":
        results, grid_coordenates = AP3("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)    
    if structure == "AP4":
        results, grid_coordenates = AP4("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)  


    
    return results, grid_coordenates, data

def losue(seed,eps,cell_size,structure,p,data,data_info,result_dir,num_points,bp,w,count):

    result_dir.mkdir(parents=True, exist_ok=True)

    auxiliar_data = result_dir / "auxiliar_data/"

    auxiliar_data.mkdir(parents=True, exist_ok=True)


    columns_name = ['seed','method','structure','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points']

    results_partial = []
    
    results_partial,grid_coordenates, data = losue_execution(results_partial,seed,eps,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info,num_points,p[17])
            
    
    csv_file_name = str(count) + "_" + str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "_LOSSUE.csv"
    data_file_name = str(cell_size) + "_" + str(structure) + "LOSSUE_data.pkl"
    grid_coordenates_file_name = str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "LOSSUE_grid.pkl"

    path_csv = result_dir / csv_file_name

    path_data = auxiliar_data / data_file_name
    
    path_grid = auxiliar_data / grid_coordenates_file_name
    
    with open(path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the column names
        writer.writerow(columns_name)
        
        # Write the data
        writer.writerows(results_partial)

    with open(path_data, 'wb') as f:
        pickle.dump(data, f)

    # with open(path_grid, 'wb') as f:
    #     pickle.dump(grid_coordenates, f)

    
def rappor(seed,eps,cell_size,structure,p,data,data_info,result_dir,num_points,count):

    result_dir.mkdir(parents=True, exist_ok=True)

    auxiliar_data = result_dir / "auxiliar_data/"

    auxiliar_data.mkdir(parents=True, exist_ok=True)

    columns_name = ['seed','method','structure','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points']

    results_partial = []

    bp = 0
    w = 1
    results_partial,grid_coordenates, data = rappor_execution(results_partial,seed, eps,bp,w,cell_size, structure, p[6][0], p[7],p[8], data, data_info,num_points,p[17])
    # else:
        #     if structure == "AP1": # w=1, use bp
        #         for bp in p[2]:
        #             w = 1
        #             results_partial,grid_coordenates, data = rappor_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #             progress_bar2.update(1)
        #     if structure == "AP2": # use w and bp
        #         for bp in p[2]:
        #             for w in p[13]:
        #                 results_partial,grid_coordenates, data = rappor_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #                 progress_bar2.update(1)
        #     else: #use w and does not use bp
        #         bp = 0
        #         for w in p[13]:
        #             results_partial,grid_coordenates, data = rappor_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #             progress_bar2.update(1)

    csv_file_name = str(count) + "_" + str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "_RAPPOR.csv"
    data_file_name = str(cell_size) + "_" + str(structure) + "RAPPOR_data.pkl"
    grid_coordenates_file_name = str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "_RAPPOR_grid.pkl"

    path_csv = result_dir / csv_file_name

    path_data = auxiliar_data / data_file_name
    
    path_grid = auxiliar_data / grid_coordenates_file_name

    with open(path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the column names
        writer.writerow(columns_name)
        
        # Write the data
        writer.writerows(results_partial)

    with open(path_data, 'wb') as f:
        pickle.dump(data, f)

    # with open(path_grid, 'wb') as f:
    #     pickle.dump(grid_coordenates, f)
        

def loloha(seed,eps,cell_size,structure,p,data,data_info,result_dir,num_points,count):

    result_dir.mkdir(parents=True, exist_ok=True)

    auxiliar_data = result_dir / "auxiliar_data/"

    auxiliar_data.mkdir(parents=True, exist_ok=True)

    columns_name = ['seed','method','structure','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points']

    results_partial = []

    bp = 0
    w = 1
    results_partial,grid_coordenates, data = loloha_execution(results_partial,seed, eps,bp,w,cell_size, structure, p[6][0], p[7],p[8], data, data_info,num_points,p[17])
        # else:
        #     if structure == "AP1": # w=1, use bp
        #         for bp in p[2]:
        #             w = 1
        #             results_partial,grid_coordenates, data = loloha_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #             progress_bar2.update(1)
        #     if structure == "AP2": # use w and bp
        #         for bp in p[2]:
        #             for w in p[13]:
        #                 results_partial,grid_coordenates, data = loloha_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #                 progress_bar2.update(1)
        #     else: #use w and does not use bp
        #         bp = 0
        #         for w in p[13]:
        #             results_partial,grid_coordenates, data = loloha_execution(results_partial, p[0], eps_perm,bp,w,cell_size, structure, p[6][0], p[7], p[8],data,data_info)
        #             progress_bar2.update(1)

    csv_file_name = str(count) + "_" + str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "_LOLOHA.csv"
    data_file_name = str(cell_size) + "_" + str(structure) + "LOLOHA_data.pkl"
    grid_coordenates_file_name = str(cell_size) + "_" + str(structure) + "_bp_" +str(bp) + "_w_" + str(w) + "_seed_" + str(seed) + "_nump_"+ str(num_points) + "_LOLOHA_grid.pkl"

    path_csv = result_dir / csv_file_name

    path_data = auxiliar_data / data_file_name
    
    path_grid = auxiliar_data / grid_coordenates_file_name
    
    with open(path_csv, 'w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the column names
        writer.writerow(columns_name)
        
        # Write the data
        writer.writerows(results_partial)

    with open(path_data, 'wb') as f:
        pickle.dump(data, f)

    # with open(path_grid, 'wb') as f:
    #     pickle.dump(grid_coordenates, f)

#########################################################################################
#########################################################################################
#########################################################################################
######################## Load Parameters ################################################
#########################################################################################

def parse_parameters(line):
    parameters = {}
    pairs = line.strip().split(';')
    for pair in pairs:
        key, value = pair.split(':')
        
        values_vector = []
        for values in value.split(','):
            values_vector.append(values)      
        
        parameters[key] = values_vector
    return parameters

def get_parameters(experiment):
                            
    p = []
    p.append(0.4) # alpha - p[0]
    p.append([float(x) for x in experiment['e']]) # eps_perm - p[1]
    p.append([float(x) for x in experiment['e_prop']]) # budget_proportion - p[2]
    p.append([int(x) for x in experiment['g']]) # cell_size - p[3]
    p.append(["Uniform","AP2","AP3","AP4"]) # structure - p[4]
    # p.append(["AP1"])
    p.append(1) # nb_seed - p[5]
    p.append([0.01]) # fr - p[6]
    p.append(True) # check_norm - p[7]
    p.append(True) # tr_count - p[8]
    p.append(None) # data_set_type - p[9]
    p.append(None) # data_set_distribution - p[10]
    #p.append(Path().resolve()/ "Dataset/Geolife_Trajectories_Dataset/Taxi/geolife_cartesian_bounded_20_10s.pkl") #p[11]
    p.append(Path().resolve()/ "Dataset/Taxi_Porto_KAGGLE/new_taxi_portugal_10000_20.pkl")
    p.append(experiment['folder'][0]) #p[12]
    p.append([int(x) for x in experiment['w']]) #p[13]
    p.append(["LOSUE","LOLOHA","RAPPOR"]) #p[14]
    p.append([int(x) for x in experiment['p']]) #p[15]
    p.append(1000) # Usuarios - p[16]
    p.append(False) # normsub - p[17]
    
    return p    



def generate_data(data_type,speed,num_users,num_points,file_path,file_info,syntetic_path):

#############################################################
#                     Generate Data                         #
#############################################################

    ti = speed #seconds

    # to load syntetic data from file
    if data_type == "l":
        #need fix to get information hardcoded in the generation

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            # x_min, y_min, x_max, y_max, num_points, n_users  = get_data_boundaries(data)
            print("Sucefully loaded user_data with len:",len(data))

        with open(file_info, mode='r') as file:
            reader = csv.reader(file)
            # Skip the header
            next(reader)
            # Read the data
            for row in reader:
                x_min, x_max, y_min, y_max = map(int, row)
    
    elif data_type == 'u' or data_type == 'n':
        #loading pre_saved syntetic data
        data, x_min, y_min, x_max, y_max = load_data(syntetic_path)
        data = data[:num_users]

        #generating a new syntetic data
        # if data_type == 'u':
        #     data, x_min, y_min, x_max, y_max = gen_execution_data('s','u',syntetic_path,ti,num_points,num_users,speed)
        # else:
        #     data, x_min, y_min, x_max, y_max = gen_execution_data('s','n',syntetic_path,ti,num_points,num_users,speed)
    
       
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        with open(file_info, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['x_min', 'x_max', 'y_min', 'y_max'])
            # Write the data
            writer.writerow([x_min, x_max, y_min, y_max])

        print("Sucefully saved user_data!")

    else:
        # print(syntetic_path)
        data, x_min, y_min, x_max, y_max = gen_execution_data('r','n',syntetic_path,ti,num_points,num_users,speed)

    return data, x_min, y_min, x_max, y_max


def continue_execution(folder,execution_list):
    # Caminho da pasta onde os arquivos estão localizados
    
    # Lista para armazenar os números extraídos
    numeros_extraidos = []

    # Itera pelos arquivos na pasta
    for arquivo in os.listdir(folder):
        if arquivo.endswith(".csv"):  # Verifica se é um arquivo CSV
            try:
                # Extrai o número antes do primeiro _
                numero = int(arquivo.split('_')[0])
                numeros_extraidos.append(numero)
            except ValueError:
                pass  # Ignora arquivos que não seguem o padrão

    numeros_extraidos = sorted(numeros_extraidos)

    
    # Crie a lista de índices que devem ser mantidos (os ausentes)
    new_list = [execution_list[i] for i in range(len(execution_list)) if i not in numeros_extraidos]
    
    return new_list



# Função auxiliar para executar cada função com seus argumentos
def executar_task(task):
    func, args = task
    return func(*args)
   

def process_sample(experiment,result_dir,data, x_min, y_min, x_max, y_max):


    ##### PARAMETERS SETUP #####
    
    # results_partial = pd.DataFrame(columns=['budget','budget_proportion','budget_base','cell_size','method','mse','mae','structure','seed','sample','grid_size','num_users','tr'])
    p = get_parameters(experiment)

    print("alpha:",p[0])
    print("eps_perm:",p[1])
    print("budget_proportion:",p[2])
    print("cell_size:",p[3])
    print("structure:",p[4])
    print("nb_seed:",p[5])
    print("fr:",p[6])
    print("check_norm:",p[7])
    print("tr_count:",p[8])
    print("data_set_type:",p[9])
    print("data_set_distribution:",p[10])
    print("data_set_path:",p[11])
    print("name:",p[12])
    print("w:",p[13])
    print("methods:",p[14])
    print("num_points:",p[15])
    print("n_users:",p[16])



    n_users = p[16]
    
    #############################################################
    #                     EXECUTION                             #
    #############################################################


    result_dir = result_dir / p[12]
   
    result_dir.mkdir(parents=True, exist_ok=True)

    profile_filename = result_dir / 'profile.csv'

    # Convert nested lists into strings
    processed_data = [str(item) if isinstance(item, (list, bool, str)) else item for item in p]

    with open(profile_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'Value'])  # Add a header row
        for index, value in enumerate(processed_data):
            writer.writerow([index, value])

    print(f"Profile saved at: {result_dir}")



    # Create a list of tasks based on combinations of e, c, m, and s
    tasks = []
    
    max_processes = multiprocessing.cpu_count()

    count = 0


    # ##############################################################

    for num_points in p[15]:
        
        data_info = (x_min, y_min, x_max, y_max, num_points, n_users)
        # Use list comprehension to truncate each inner list to the first `n` elements
        new_data = [inner_list[:num_points] for inner_list in data]
        # x_min, y_min, x_max, y_max, num_points, n_users  = get_data_boundaries(new_data)
        # data_info = (x_min, y_min, x_max, y_max, num_points, n_users)
            
    
        for seed in range(p[5]):
            for e in p[1]:
                for c in p[3]:
                    for m in p[14]:
                        for s in p[4]:
                            if m == "LOSUE":
                                if s == "Uniform":
                                    tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,0,1,count)))
                                    count += 1
                                else:
                                    if s == "AP1":
                                        for bp in p[2]:
                                            tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,bp,1,count)))
                                            count += 1
                                    else:
                                        for w in p[13]:
                                            if s=='AP2':
                                                for bp in p[2]:
                                                    tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,bp,w,count)))
                                                    count += 1    
                                            else:
                                                tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,0,w,count)))
                                                count += 1
                            else:
                                if m == "LOLOHA" and s == "Uniform":
                                    tasks.append((loloha, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,count)))
                                    count += 1
                                else:
                                    if m == "RAPPOR" and s == "Uniform":
                                        tasks.append((rappor, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,count)))
                                        count += 1

    tasks = continue_execution(result_dir, tasks)
    
    print("Number of counts:",count)
    print("max process:",max_processes)
    print("Number of tasks:",len(tasks))
    
    # Use multiprocessing with tqdm to show progress
    
    with Pool(processes=(max_processes-1)) as pool:
        
        #list(tqdm(pool.map(executar_task, tasks), total=len(tasks), desc="Executando tasks"))
        for _ in tqdm(pool.imap_unordered(executar_task, tasks), total=len(tasks), desc="Executando tasks"):
            pass


# ########################## EXECUTION WITH THREADS #####################################


def main():

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python3 program.py <filename>")
        sys.exit(1)

    # Access the argument (filename in this case)
    filename = sys.argv[1]

    # Print or process the file name
    print(f"Received file name: {filename}")
    
    # Example: Open and read the file
    try:
        with open(filename, 'r') as file:
            content = file.read()
            print("File content:")
            print(content)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)


    # print("####Starting####")
    exp_starttime = time.time()

    experiments = []

    # with open('testes_profile_exp12_simple_porto.txt', 'r') as file:
    #     for line in file:
    #         # print(line)
    #         parameters = parse_parameters(line)
    #         experiments.append(parameters)

    with open(filename, 'r') as file:
        for line in file:
            # print(line)
            parameters = parse_parameters(line)
            experiments.append(parameters)
    

    result_dir = Path().resolve() / "Results" / experiments[1]['name'][0] / str(datetime.date.today())
    # result_dir = Path().resolve() / "Results" / experiments[1]['name'][0] / '2025-01-07'

    file_path = result_dir / "data.pkl"
    file_info = result_dir / "data_info.csv"

    # print(result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    data_type = experiments[0]['d'][0]

    speed = 40
    num_users = 1000
    num_points = 40

    if data_type == 'g':
        syntetic_path = Path().resolve() / "Dataset/Geolife_Trajectories_Dataset/Taxi/geolife_cartesian_bounded_20_10s.pkl"
    elif data_type == 'p':
        syntetic_path = Path().resolve() / "Dataset/Taxi_Porto_KAGGLE/new_taxi_portugal_10000_20.pkl"
    elif data_type == 'u':
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Uniform/data_uni.pkl"
    elif data_type == 'n':
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Normal/data_norm.pkl"
    else:
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Direction/data_dir.pkl"

    data, x_min, y_min, x_max, y_max = generate_data(data_type,speed,num_users,num_points,file_path,file_info,syntetic_path)

    # print(len(data))
    # print(x_min)
    # print(x_max)
    # print(y_min)
    # print(y_max)

    # Print the parsed experiments
    print("Number of experiments:", int(experiments[0]['n'][0]))
    print("Number of seeds:", int(experiments[0]['s'][0]))
    print("Data Type:", data_type)
    print("Data_Path:",syntetic_path)


    for i in range(2,len(experiments)):
        print(f"Experiment {i-1}: {experiments[i]}")
        print(experiments[i]['folder'][0])
        process_sample(experiments[i],result_dir,data, x_min, y_min, x_max, y_max)



    # for setup in range(len(experiments)-1):
    #     process_sample(experiments[setup+1])
    #     process_sample(experiments[2])


if __name__ == "__main__":
    main()

