########################################################################################################
#
#   Experiment 14a: limited Merge to 1 (one merge in first interection) and at most deep x in split     #
#   Base Line: Uniform QuadTree                                                                         #
#   Two Round Execution:                                                                                #
#   AP2: - Adaptative Grip with 1 splits after x timestamp, using a the previous grid as base           #
#        - The split has two executions rounds, the first one with a budget of bp and the               #
#        - second one with a budget of 1-bp where the report will be collected based on the new grid.   #
#   One Round Execution:                                                                                #
#   AP3: - Adaptative Grip with 1 splits after x timestamp, using the previous grid as base             #
#        - The split has one execution only. THe new grid will be used only in the next timestamp.      #
#   AP4: - Adaptative Grip with 1 splits after x timestamp, using a fixed size grid as base             #
#        - The split has one execution only. THe new grid will be used only in the next timestamp.
# 
#   ALL The Approaches will be executed with the following protocols: LOSUE                             #
#########################################################################################################

# with adaptative w

import multiprocessing
import csv
## Structure imports

from shapely.geometry import Point
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import datetime
from multiprocessing import Pool
import os
from shapely.ops import unary_union
import copy
import pickle
import sys
import math

from dataset import load_data

## loloha imports
import warnings; warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.spatial.distance import cosine
from quadtreev3 import QuadTree
from grid import Grid


from LOLOHA_MOD.LDP.protocols import RAPPOR_Client_TAU, RAPPOR_Aggregator # [1]
from LOLOHA_MOD.LDP.protocols import L_GRR_Client_TAU, L_GRR_Aggregator 
from LOLOHA_MOD.LDP.protocols import L_OSUE_Client_TAU, L_OSUE_Aggregator
from LOLOHA_MOD.LDP.protocols import LOLOHA_Client_TAU, LOLOHA_Aggregator_TAU 

#####################################################################
#                           Auxiliar Functions                      #
#####################################################################

# import sys
sys.setrecursionlimit(200000000)  # Ajuste conforme necessário


def split_grid_privag(grid,e,a,est_freq,num_users):
    

    grid_list = []
    for i in range(len(grid)):
        grid_list.append([grid.iloc[i]['geometry'].bounds,grid.iloc[i]['label'],est_freq[i]])
    
    new_grid_list = []
    
    for cell in grid_list:
        numerator = 2 * a * cell[2] * (math.exp(e) - 1) * math.sqrt(cell[2]*num_users)
        denominator = math.exp(e)
        value = numerator / denominator
        g = int(math.sqrt(value))
        
        
        if g > 1:
            # print("g:",g)
            new_grid_list += split(cell,g)
        else:
            new_grid_list.append(cell)

            
    x_min,y_min,x_max,y_max = 0,0,0,0
    count = 0

    # print("GRID:",grid_list[0])

    new_grid_instance = Grid((x_min,y_min),(x_max,y_max),count)
    new_grid_instance.convert_grid_list(new_grid_list)

    return new_grid_instance.grid


def split(cell,g):
    
    new_cells = []
    new_fr = cell[2] / (g*g)
    
    xmin = cell[0][0]
    ymin = cell[0][1]
    xmax = cell[0][2]
    ymax = cell[0][3]

    granularity = (xmax - xmin)/g


    for i in range(g):
        for j in range(g):
            x_min = xmin  + j * granularity
            y_min = ymin + i * granularity
            x_max = x_min + granularity
            y_max = y_min + granularity
                
            new_cell = [[x_min,y_min,x_max,y_max],cell[1],new_fr]
                    
            new_cells.append(new_cell)   

        
    return new_cells

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

    # print("GRID:",grid_list[0])

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



def encode_dada(data,qtree):
    enc_data = []

    for i in range(len(data)):
        enc_locations = []
        for tau in range(len(data[i])):
            P = (data[i][tau][0],data[i][tau][1])
        
            id = qtree.find_leaf(*P).id
             
            enc_locations.append(id)
        enc_data.append(enc_locations)

    return enc_data

def encode_dada_grid(data,grid):
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


def gen_execution_data(file_name):
            
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
############################### AP2 #################################
#####################################################################

def AP2(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    execution_times = 2
  
    x_min, y_min, x_max, y_max, num_points, n_users = data_info

    tau = num_points

    eps_perm_base = eps_perm

    # alpha = 1/tau

    alpha = 0.1

    num_users = len(data)
        
    # eps_perm_base = eps_perm

    # eps_perm_per_tau = eps_perm_base / num_points
    
    quad_tree_base = QuadTree(x_min,x_max,y_min,y_max,cell_size)
    quad_tree_base.build()
    quad_tree_base.reset_ids_leafs()

    quad_tree = copy.deepcopy(quad_tree_base)

    map_vector = quad_tree.leaf_ids()
    
    tr_value = 1
    
        
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    split_window = []
    similarity_vector = []
    
    # client - side
    
    k =  len(map_vector)

    grid_size_base = k

    eps_perm_used = 0

    
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

    split_flag = 1
    new_grid = 1
    cosine_similarity = 0
    old_est_freq = np.zeros(k)
    w_vector = []
    w_count = 0
    rolling_window = 3
    freq_window = []

    eps_perm_used = 0

    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        
        tr_vector = []
        
        k = len(map_vector)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,quad_tree)   

        if protocol == 'LOLOHA':
            reduction_factor = 0.6
        else:
            reduction_factor = 0.1

        if t == 0 or split_flag:
            eps_perm_used = eps_perm_base
        else:
            eps_perm_used = eps_perm_base * reduction_factor



        # if t % w == 0:
        #     split_flag = 1
        #     new_grid = 1
        # else:
        #     split_flag = 0
        #     new_grid = 0
        #     old_est_freq = est_freq

        #### 0, 5,10,15,20
        if t == 0 or cosine_similarity > 0.7:
            # print("cosine_similarity:",cosine_similarity)
            # print("t:",t)
            split_flag = 1
            new_grid = 1
            w_vector.append(w_count)
            w_count = 0
            freq_window = []
        else:   
            split_flag = 0
            new_grid = 0
            # usada somente na proxima vez que for calcular a cosine similarity
            # old_est_freq = est_freq
            w_count += 1

        if t == tau - 1:
            w_vector.append(w_count)
        
        
        #### 0, 5,10,15,20
        if split_flag:
            # print("Spliting")
            execution_times = 2
            eps_perm1 = eps_perm_used * bp
            eps_perm2 = eps_perm_used * (1-bp)
            eps_11 = eps_perm1 * alpha
            eps_12 = eps_perm2 * alpha
        else:   
            execution_times = 1
            eps_perm1 = eps_perm_used
            eps_perm2 = eps_perm_used
            eps_11 = eps_perm1 * alpha
            eps_12 = eps_perm2 * alpha

        
        for e in range(execution_times):

            reports = []
            
            if e != 0:
                memoization = False
               
                if split_flag:
                    # print("k:",k)
                    if tr_count:
                        tr_value = max(int(n_users/k),30)
                        # tr_value = max(int(10000/k),30)
                        # tr_value = int(10000 * tr)
                        tr_merge = min(int(tr_value/4),10)
                        #tr_value = int(n_users/k) # looking for uniformity
                    else:
                        tr_value = 1 / k

                mapeamento = quad_tree.get_map(k)

                quad_tree.set_counts(count_vector)
                quad_tree.calculate_repo_counts()

                quad_tree.naive_split(tr_value,mapeamento,k,2)
                quad_tree.calculate_repo_counts()
                quad_tree.reset_ids_leafs()

                if t == 0:
                    quad_tree.merge_children_drop_child(tr_merge)
                    quad_tree.calculate_repo_counts()
                    quad_tree.reset_ids_leafs()

                # quad_tree.naive_split(tr_value)
                # quad_tree.merge_children(tr_merge,mapeamento)
                # quad_tree.merge_children_drop_child(tr_merge)
                # quad_tree.merge_children(tr_merge)

                map_vector = quad_tree.leaf_ids()
                
                k = len(map_vector)
                data_encoded = encode_dada(data_tau,quad_tree)

                eps_perm_temp = eps_perm2
                eps_1_temp = eps_12

                if protocol == 'LOLOHA':
                    g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
                                      - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
                                        + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
                                          + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
                                            / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                    user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
                else:
                    user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]
                
                # memoization = False

            else:
                memoization = True
                eps_perm_temp = eps_perm1
                eps_1_temp = eps_11

                # memoization = True

            hits = 0
            for i in range(n_users):
                if protocol == 'RAPPOR':
                    report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LGRR':
                    report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOSUE':
                    report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOLOHA':
                    report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                # print("budget_used:",budget_used,eps_perm_used,eps_perm_base)

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

            count_vector = [int(f * 10000) for f in est_fr_vector]

        
        

        hit_vector.append(hits)
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
        
        mse = mean_squared_error(real_freq*10000, est_freq*10000)

        if w_count < rolling_window:
            freq_window.append(est_freq)
            old_est_freq = est_freq
        else:
            old_est_freq = np.mean(freq_window,axis=0)
            freq_window.insert(0,est_freq)
            freq_window.pop()

        cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))
        # old_est_freq = est_freq
        
        similarity_vector.append(cosine_similarity)
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))

    end_time = time.time()

    run_time = end_time - start_time

    budget_tracking = np.mean(final_budget_users) 

    #get a integer value for w_vector mean
    w = int(np.mean(w_vector))

    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,run_time,np.mean(hit_vector)))
    

    return results, []


#####################################################################
############################### AP3 #################################
#####################################################################

def AP3(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    # alpha = 1/tau
    alpha = 0.1

    num_users = len(data)

    quad_tree = QuadTree(x_min,x_max,y_min,y_max,cell_size)
    quad_tree.build()
    quad_tree.reset_ids_leafs()

    
    map_vector = quad_tree.leaf_ids()
    
    tr_value = 1
    
        
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    similarity_vector = []
    
    # client - side
    
    k =  len(map_vector)

    guad_size_base = k

    if protocol == 'LOLOHA':
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]  

    final_budget_users = [0 for _ in range(n_users)]


    split_flag = 1
    new_grid = 1
    old_est_freq = np.zeros(k)
    cosine_similarity = 0
    w_vector = []
    w_count = 0
    rolling_window = 3
    freq_window = []
    eps_perm_used = 0

    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        tr_vector = []
        
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,quad_tree)   

        
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        if protocol == 'LOLOHA':
            reduction_factor = 0.6
        else:
            reduction_factor = 0.1

        if t == 0 or split_flag:
            eps_perm_used = eps_perm_base
        else:
            eps_perm_used = eps_perm_base * reduction_factor

        eps_1 = eps_perm_used * alpha

        memoization = True

        # #### 0, 5,10,15,20
        # if mse > 0.01:
        #     grid_reconstrution = True
        #     split_window.append(t)
        # else:   
        #     grid_reconstrution = False

        reports = []
            
           
        hits = 0
        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_used, eps_1,user_memo_vector[i], memoization)


            reports.append(report)
            final_budget_users[i] += budget_used


        ts_values = []
        for loc in data_encoded:
            ts_values.append(loc[0])
    
        # Server-Side

        real_freq = get_real_freq(ts_values,k)
        if protocol == 'RAPPOR':
            est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm_used, eps_1,notmsub)
        if protocol == 'LGRR':
            est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm_used, eps_1,notmsub)
        if protocol == 'LOSUE':
            est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm_used, eps_1,notmsub)
        if protocol == 'LOLOHA':
            est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm_used, eps_1, g,notmsub)
    
        est_fr_vector = [fr for fr in est_freq]

        count_vector = [int(f * 10000) for f in est_fr_vector]

        hit_vector.append(hits)
        tr_vector.append(tr_value)

        est_frequency_vector.append(est_fr_vector)     
            
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)

        real_freq = np.nan_to_num(real_freq)
        est_freq =np.nan_to_num(est_freq)
        
        mse = mean_squared_error(real_freq*10000, est_freq*10000)

        if w_count < rolling_window:
            freq_window.append(est_freq)
            old_est_freq = est_freq
        else:
            old_est_freq = np.mean(freq_window,axis=0)
            freq_window.insert(0,est_freq)
            freq_window.pop()
            
        cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))
        
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))
        similarity_vector.append(cosine_similarity)


         #### 0, 5,10,15,20
        if t == 0 or cosine_similarity > 0.7:
            # print("cosine_similarity:",cosine_similarity)
            # print("t:",t)
            split_flag = 1
            new_grid = 1
            w_vector.append(w_count)
            w_count = 0
            freq_window = []
        else:   
            split_flag = 0
            new_grid = 0
            # usada somente na proxima vez que for calcular a cosine similarity
            # old_est_freq = est_freq
            w_count += 1

        if t == tau - 1:
            w_vector.append(w_count)
            
        # if t % w == 0:
        #     split_flag = 1
        #     new_grid = 1
        #     split_window.append(t)
        # else:
        #     split_flag = 0
        #     new_grid = 0
        #     old_est_freq = est_freq

        ### Reconstrution ###
        if split_flag:
            # print("k:",k)
            if tr_count:
                tr_value = max(int(n_users/k),30)
                # tr_value = max(int(10000/k),30)
                # tr_value = int(10000 * tr)
                tr_merge = min(int(tr_value/4),10)
            
            # print("counting vector:",sum(count_vector))
            
            mapeamento = quad_tree.get_map(k)

            quad_tree.set_counts(count_vector)

            quad_tree.naive_split(tr_value,mapeamento,k,2)
            quad_tree.calculate_repo_counts()
            quad_tree.reset_ids_leafs()

            if t == 0:
                quad_tree.merge_children_drop_child(tr_merge)
                quad_tree.calculate_repo_counts()    
                quad_tree.reset_ids_leafs()
            
            
            map_vector = quad_tree.leaf_ids()
            
            k = len(map_vector)
            data_encoded = encode_dada(data_tau,quad_tree)

            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]

    end_time = time.time()

    running_time = end_time - start_time

    budget_tracking = np.mean(final_budget_users) 

    #get a integer value for w_vector mean
    w = int(np.mean(w_vector))

    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,guad_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,running_time, np.mean(hit_vector)))
    

    return results, []

def Alog(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):

    # ti = 60 #seconds
    execution_times = 2
    # jump = 5 * 60 / ti #(every 5 minutes we will change the base grid)

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points

    eps_perm_base = eps_perm

    alpha = 0.1

    num_users = len(data)
        
    # eps_perm_base = eps_perm

    # eps_perm_per_tau = eps_perm_base / num_points
    
    # if cell_size == 2200:
    #     cell_size = 2700
    # if cell_size == 700:
    #     cell_size = 1380
    # if cell_size == 400:
    #     cell_size = 2700

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
    hit_vector = []
    split_window = []
    similarity_vector = []
    
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

    split_flag = 1
    new_grid = 1
    
    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)

        k = len(grid)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada_grid(data_tau,grid)   

        if t % w == 0:
            split_flag = 1
            new_grid = 1
        else:
            split_flag = 0
            new_grid = 0
            old_est_freq = est_freq

        
        #### 0, 5,10,15,20
        if split_flag:
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
                    # tr_value = int(10000 * tr)
                    tr_value = int(max(n_users/k,30)) # looking for uniformity
                else:
                    tr_value = 1 / k
                grid_base = split_grid(grid, est_freq, num_users, tr_value, check_norm,tr_count)

                k = len(grid_base)
                data_encoded = encode_dada_grid(data_tau,grid)

                eps_perm_temp = eps_perm2
                eps_1_temp = eps_12

                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]
                
                # memoization = False

            else:
                memoization = True
                eps_perm_temp = eps_perm1
                eps_1_temp = eps_11

                # memoization = True

            hits = 0
            for i in range(n_users):
                if protocol == 'RAPPOR':
                    report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LGRR':
                    report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOSUE':
                    report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOLOHA':
                    report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)


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

        hit_vector.append(hits)
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
        
        mse = mean_squared_error(real_freq, est_freq)

        if new_grid:
            old_est_freq = est_freq

        cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))
        

        # if  cosine_similarity > 0.001:
        #     split_flag = 1
        #     new_grid = 1
        #     split_window.append(t)
        # else:
        #     split_flag = 0
        #     new_grid = 0
        #     old_est_freq = est_freq


        
        similarity_vector.append(cosine_similarity)
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq, est_freq))
    
    end_time = time.time()

    running_time = end_time - start_time

    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,running_time, np.mean(hit_vector)))
    

    return results, grid_coordenates

def Alog_quad(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    execution_times = 2
  
    x_min, y_min, x_max, y_max, num_points, n_users = data_info

    tau = num_points

    eps_perm_base = eps_perm

    # alpha = 1/tau

    alpha = 0.1

    quad_tree_base = QuadTree(x_min,x_max,y_min,y_max,cell_size)
    quad_tree_base.build()
    quad_tree_base.reset_ids_leafs()

    quad_tree = copy.deepcopy(quad_tree_base)

    map_vector = quad_tree.leaf_ids()
    
    tr_value = 1
    
        
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    split_window = []
    similarity_vector = []
    
    # client - side
    
    k =  len(map_vector)

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

    split_flag = 1
    new_grid = 1
    
    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        
        tr_vector = []
        
        k = len(map_vector)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,quad_tree)   

        if t % w == 0:
            split_flag = 1
            new_grid = 1
        else:
            split_flag = 0
            new_grid = 0
            old_est_freq = est_freq

        
        if split_flag:
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
               
                if split_flag:
                    # print("k:",k)
                    if tr_count:
                        tr_value = max(int(n_users/k),30)
                        
                    else:
                        tr_value = 1 / k

                mapeamento = quad_tree.get_map(k)

                quad_tree.set_counts(count_vector)
                quad_tree.calculate_repo_counts()

                quad_tree.naive_split(tr_value,mapeamento,k,2)
                quad_tree.calculate_repo_counts()
                quad_tree.reset_ids_leafs()

               
                map_vector = quad_tree.leaf_ids()
                
                k = len(map_vector)
                data_encoded = encode_dada(data_tau,quad_tree)

                eps_perm_temp = eps_perm2
                eps_1_temp = eps_12

                if protocol == 'LOLOHA':
                    g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
                                      - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
                                        + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
                                          + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
                                            / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                    user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
                else:
                    user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]
                
                # memoization = False

            else:
                memoization = True
                eps_perm_temp = eps_perm1
                eps_1_temp = eps_11

                # memoization = True


            hits = 0
            for i in range(n_users):
                if protocol == 'RAPPOR':
                    report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LGRR':
                    report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOSUE':
                    report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                if protocol == 'LOLOHA':
                    report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_temp, eps_1_temp,user_memo_vector[i], memoization)
                # print("budget_used:",budget_used,eps_perm_used,eps_perm_base)

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

            count_vector = [int(f * 10000) for f in est_fr_vector]

        
        
        hit_vector.append(hits)
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
        
        mse = mean_squared_error(real_freq*10000, est_freq*10000)

        if new_grid:
            old_est_freq = est_freq

        cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))
        # old_est_freq = est_freq
        
        similarity_vector.append(cosine_similarity)
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))

    end_time = time.time()

    run_time = end_time - start_time

    budget_tracking = np.mean(final_budget_users) 

    
    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,grid_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,run_time, np.mean(hit_vector)))
    

    return results, []



# def AP4(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
#     print("ALOG_1Rb:",cell_size)
#     x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
#     tau = num_points
        
#     eps_perm_base = eps_perm

#     alpha = 1/tau

#     print("alpha:",alpha)
#     print("tau:",tau)
#     print("eps_base:",eps_perm_base)   

#     num_users = len(data) 
    
#     quad_tree_base = QuadTree(x_min,x_max,y_min,y_max,cell_size)
#     quad_tree_base.build()
#     quad_tree_base.reset_ids_leafs()

#     quad_tree = copy.deepcopy(quad_tree_base)

#     map_vector = quad_tree.leaf_ids()
    
#     tr_value = 1
    
        
#     lst_mse = [] # List of all MSE per data collection
#     lst_mae = [] # List of all MAE per data collection
#     grid_k = []
#     tr_vector = []
#     est_frequency_vector = []
#     similarity_vector = []
    
#     # client - side
    
#     k =  len(map_vector)

#     quad_size_base = k

#     if protocol == 'LOLOHA':
#         g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm)
#                                       - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1))
#                                         + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm)
#                                           + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1)
#                                             / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
#         user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
#     else:
#         user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]   

#     final_budget_users = [0 for _ in range(n_users)]

#     split_flag = 1
#     new_grid = 1
#     old_est_freq = np.zeros(k)
#     cosine_similarity = 0
#     w_vector = []
#     w_count = 0
#     rolling_window = 3
#     freq_window = []

#     for t in range(tau): # For each data collection
#         # print("Execution of ts:",t)
#         count_vector = []
#         tr_vector = []
        

        
#         data_tau = [[lista[t]] for lista in data]
#         data_encoded = encode_dada(data_tau,quad_tree)   

        
#         # eps_perm = eps_perm_per_tau
#         # eps_1 = eps_perm_per_tau * alpha

#         eps_1 = eps_perm * alpha

#         memoization = True

        

#         reports = []
            
    
#         for i in range(n_users):
#             if protocol == 'RAPPOR':
#                 report, user_memo_vector[i], budget_used = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
#             if protocol == 'LGRR':
#                 report, user_memo_vector[i], budget_used = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
#             if protocol == 'LOSUE':
#                 report, user_memo_vector[i], budget_used = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
#             if protocol == 'LOLOHA':
#                 report, user_memo_vector[i], budget_used = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)


#             reports.append(report)
#             final_budget_users[i] += budget_used


#         ts_values = []
#         for loc in data_encoded:
#             ts_values.append(loc[0])
    
#         # Server-Side

#         real_freq = get_real_freq(ts_values,k)
#         if protocol == 'RAPPOR':
#             est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
#         if protocol == 'LGRR':
#             est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm, eps_1,notmsub)
#         if protocol == 'LOSUE':
#             est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm, eps_1,notmsub)
#         if protocol == 'LOLOHA':
#             est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm, eps_1, g,notmsub)
    
#         est_fr_vector = [fr for fr in est_freq]

#         count_vector = [int(f * 10000) for f in est_fr_vector]

#         tr_vector.append(tr_value)

#         est_frequency_vector.append(est_fr_vector)     
            
#         grid_k.append(k) #Save the list of grids used
#                                 # tr_tau.append(tr)
#         mse = mean_squared_error(real_freq*10000, est_freq*10000)
        
#         if w_count < rolling_window:
#             freq_window.append(est_freq)
#             old_est_freq = est_freq
#         else:
#             old_est_freq = np.mean(freq_window,axis=0)
#             freq_window.insert(0,est_freq)
#             freq_window.pop()
            
#         cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))

#         real_freq = np.nan_to_num(real_freq)
#         est_freq =np.nan_to_num(est_freq)
        
#         lst_mse.append(mse)
#         lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))
#         similarity_vector.append(cosine_similarity)


#          #### 0, 5,10,15,20
#         if t == 0 or cosine_similarity > 0.55:
#             # print("cosine_similarity:",cosine_similarity)
#             # print("t:",t)
#             split_flag = 1
#             new_grid = 1
#             w_vector.append(w_count)
#             w_count = 0
#             freq_window = []
#         else:   
#             split_flag = 0
#             new_grid = 0
#             # usada somente na proxima vez que for calcular a cosine similarity
#             # old_est_freq = est_freq
#             w_count += 1

#         if t == tau - 1:
#             w_vector.append(w_count)

#         # if t % w == 0:
#         #     split_flag = 1
#         #     new_grid = 1
#         #     split_window.append(t)
#         # else:
#         #     split_flag = 0
#         #     new_grid = 0
#         #     old_est_freq = est_freq


#         ### Reconstrution ###
#         if split_flag:
#             # print("k:",k)
#             if tr_count:
#                 # tr_value = int(n_users * tr)
#                 tr_value = int(10000/k)
#                 # tr_value = int(10000 * tr)
#                 tr_merge = min(int(tr_value/4),10)
#                 #tr_value = int(n_users/k) # looking for uniformity
#             else:
#                 tr_value = 1 / k
            
#             quad_tree = copy.deepcopy(quad_tree_base)
#             quad_tree.set_counts(count_vector)

#             mapeamento = quad_tree.get_map(k)

#             quad_tree.naive_split(tr_value,mapeamento,k)
#             # quad_tree.naive_split(tr_value)
#             quad_tree.calculate_repo_counts()
#             # quad_tree.merge_children(tr_merge,mapeamento)
#             quad_tree.merge_children_drop_child(tr_merge)
#             quad_tree.calculate_repo_counts()
#             quad_tree.reset_ids_leafs()
#             map_vector = quad_tree.leaf_ids()

#             k = len(map_vector)
#             data_encoded = encode_dada(data_tau,quad_tree)

#             if protocol == 'LOLOHA':
#                 g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
#                 user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
#             else:
#                 user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]

            
#     budget_tracking = np.mean(final_budget_users) 

#     w = int(np.mean(w_vector))

#     results.append((seed,protocol,structure,eps_perm,eps_perm_base,bp,quad_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector))
    

#     return results, []

def AP4(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    # print("ALOG_1Rb:",cell_size)
    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    alpha = 0.1

    # print("alpha:",alpha)
    # print("tau:",tau)
    # print("eps_base:",eps_perm_base)   

    num_users = len(data) 
    
    quad_tree_base = QuadTree(x_min,x_max,y_min,y_max,cell_size)
    quad_tree_base.build()
    quad_tree_base.reset_ids_leafs()

    quad_tree = copy.deepcopy(quad_tree_base)

    map_vector = quad_tree.leaf_ids()
    
    tr_value = 1
    
        
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    similarity_vector = []
    
    # client - side
    
    k =  len(map_vector)

    quad_size_base = k

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

    split_flag = 1
    new_grid = 1
    old_est_freq = np.zeros(k)
    cosine_similarity = 0
    w_vector = []
    w_count = 0
    rolling_window = 3
    freq_window = []

    eps_perm_used = 0

    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        tr_vector = []
        

        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,quad_tree)   

        
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        if protocol == 'LOLOHA':
            reduction_factor = 0.6
        else:
            reduction_factor = 0.1


        if t == 0 or split_flag:
            eps_perm_used = eps_perm
        else:
            eps_perm_used = eps_perm * reduction_factor

        eps_1 = eps_perm_used * alpha

        memoization = True

        # print("eps_used:",eps_perm_used)

        reports = []
            
        hits = 0
        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm_used, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm_used, eps_1,user_memo_vector[i], memoization)


            reports.append(report)
            final_budget_users[i] += budget_used


        ts_values = []
        for loc in data_encoded:
            ts_values.append(loc[0])
    
        # Server-Side

        real_freq = get_real_freq(ts_values,k)
        if protocol == 'RAPPOR':
            est_freq = RAPPOR_Aggregator(np.array(reports), eps_perm_used, eps_1,notmsub)
        if protocol == 'LGRR':
            est_freq = L_GRR_Aggregator(np.array(reports), k, eps_perm_used, eps_1,notmsub)
        if protocol == 'LOSUE':
            est_freq = L_OSUE_Aggregator(np.array(reports), eps_perm_used, eps_1,notmsub)
        if protocol == 'LOLOHA':
            est_freq = LOLOHA_Aggregator_TAU(np.array(reports), k, eps_perm_used, eps_1, g,notmsub)
    
        est_fr_vector = [fr for fr in est_freq]

        count_vector = [int(f * 10000) for f in est_fr_vector]

        hit_vector.append(hits)

        tr_vector.append(tr_value)

        est_frequency_vector.append(est_fr_vector)     
            
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        mse = mean_squared_error(real_freq*10000, est_freq*10000)
        
        if w_count < rolling_window:
            freq_window.append(est_freq)
            old_est_freq = est_freq
        else:
            old_est_freq = np.mean(freq_window,axis=0)
            freq_window.insert(0,est_freq)
            freq_window.pop()
            
        cosine_similarity = np.nan_to_num(cosine(est_freq, old_est_freq))

        real_freq = np.nan_to_num(real_freq)
        est_freq =np.nan_to_num(est_freq)
        
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))
        similarity_vector.append(cosine_similarity)


         #### 0, 5,10,15,20
        if t == 0 or cosine_similarity > 0.7:
            # print("cosine_similarity:",cosine_similarity)
            # print("t:",t)
            split_flag = 1
            new_grid = 1
            w_vector.append(w_count)
            w_count = 0
            freq_window = []
        else:   
            split_flag = 0
            new_grid = 0
            # usada somente na proxima vez que for calcular a cosine similarity
            # old_est_freq = est_freq
            w_count += 1

        if t == tau - 1:
            w_vector.append(w_count)

        # if t % w == 0:
        #     split_flag = 1
        #     new_grid = 1
        #     split_window.append(t)
        # else:
        #     split_flag = 0
        #     new_grid = 0
        #     old_est_freq = est_freq


        ### Reconstrution ###
        if split_flag:
            # print("k:",k)
            if tr_count:
                tr_value = max(int(n_users/k),30)
                # tr_value = max(int(10000/k),30)
                # tr_value = int(10000 * tr)
                tr_merge = min(int(tr_value/4),10)
                #tr_value = int(n_users/k) # looking for uniformity
                # print("tr_split:",tr_value)
                # print("tr_merge:",tr_merge)
            else:
                tr_value = 1 / k
            
            quad_tree = copy.deepcopy(quad_tree_base)
            
            mapeamento = quad_tree.get_map(k)

            quad_tree.set_counts(count_vector)

            quad_tree.naive_split(tr_value,mapeamento,k,2)
            quad_tree.calculate_repo_counts()
            quad_tree.reset_ids_leafs()

            if t == 0:
                quad_tree.merge_children_drop_child(tr_merge)
                quad_tree.calculate_repo_counts()
                quad_tree.reset_ids_leafs()    
            
            map_vector = quad_tree.leaf_ids()
            

            k = len(map_vector)
            data_encoded = encode_dada(data_tau,quad_tree)

            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]

            
    end_time = time.time()

    running_time = end_time - start_time

    budget_tracking = np.mean(final_budget_users) 

    w = int(np.mean(w_vector))

    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,quad_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,running_time,np.mean(hit_vector)))
    

    return results, []



def PrivAG(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):

    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    alpha = 0.1

    num_users = len(data)

    grid_instance = Grid((x_min,y_min),(x_max,y_max),cell_size)
    
    tr_value = 1
    
    grid_instance.create_syntetic_grid()

    grid_base = grid_instance.get_grid()
    
        
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    similarity_vector = []
    
    # client - side
    
    k =  len(grid_base)

    guad_size_base = k

    if protocol == 'LOLOHA':
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]  

    final_budget_users = [0 for _ in range(n_users)]

    split_flag = 1

    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        tr_vector = []
        
        grid = copy.deepcopy(grid_base)


        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada_grid(data_tau,grid)   

        
        eps_1 = eps_perm * alpha

        memoization = True

        
        reports = []
            
        hits = 0
        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)


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

        count_vector = [int(f * 10000) for f in est_fr_vector]

        hit_vector.append(hits)
        tr_vector.append(tr_value)

        est_frequency_vector.append(est_fr_vector)     
            
        grid_k.append(k) 

        real_freq = np.nan_to_num(real_freq)
        est_freq =np.nan_to_num(est_freq)
        
        mse = mean_squared_error(real_freq*10000, est_freq*10000)

        
        lst_mse.append(mse)
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))
        

        if split_flag:
            
            #### ADAPTATION PROCESS #################
            
            a = 10
            
            grid_base = split_grid_privag(grid,eps_perm,a,est_freq,num_users)
            
            k = len(grid_base)
            data_encoded = encode_dada_grid(data_tau,grid)

            if protocol == 'LOLOHA':
                g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
                user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
            else:
                user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]
            
           
    end_time = time.time()

    running_time = end_time - start_time    

    budget_tracking = np.mean(final_budget_users) 

    
    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,guad_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,running_time, np.mean(hit_vector)))
    

    return results, []



#####################################################################
############################ Uniform ################################
#####################################################################

def Uniform(protocol,results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm=False,tr_count=False,notmsub=True):
    
    x_min, y_min, x_max, y_max, num_points, n_users = data_info
  
    tau = num_points
        
    eps_perm_base = eps_perm

    # alpha = 1/tau

    alpha = 0.1

    # eps_perm_per_tau = eps_perm_base / num_points

    # print(x_min, y_min, x_max, y_max, num_points, cell_size)
    
    quad_tree = QuadTree(x_min,x_max,y_min,y_max,cell_size)
    quad_tree.build()
    quad_tree.reset_ids_leafs()

    map_vector = quad_tree.leaf_ids()
    
    tr_value = 1
    
    
    
    lst_mse = [] # List of all MSE per data collection
    lst_mae = [] # List of all MAE per data collection
    grid_k = []
    tr_vector = []
    est_frequency_vector = []
    hit_vector = []
    similarity_vector = []
    
    # client - side
    
    k =  len(map_vector)

    old_est_freq = np.zeros(k)

    quad_size_base = k

    if protocol == 'LOLOHA':
        print("eps_lo:",eps_perm)
        g = int(max(np.rint((np.sqrt(np.exp(4*eps_perm) - 14*np.exp(2*eps_perm) - 12*np.exp(2*eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+1)) + 12*np.exp(eps_perm*(alpha+3)) + 1) - np.exp(2*eps_perm) + 6*np.exp(eps_perm) - 6*np.exp(eps_perm*alpha) + 1) / (6*(np.exp(eps_perm) - np.exp(eps_perm*alpha)))), 2))
        user_memo_vector = [{val: None for val in range(g)} for _ in range(n_users)]
    else:
        user_memo_vector = [{val: None for val in range(k)} for _ in range(n_users)]    

    final_budget_users = [0 for _ in range(n_users)]

    start_time = time.time()

    for t in range(tau): # For each data collection
        # print("Execution of ts:",t)
        count_vector = []
        tr_vector = []
        
    #     grid = copy.deepcopy(grid_base)

        k = len(map_vector)
        
        data_tau = [[lista[t]] for lista in data]
        data_encoded = encode_dada(data_tau,quad_tree)   

        
        reports = []
        
    
        # eps_perm = eps_perm_per_tau
        # eps_1 = eps_perm_per_tau * alpha

        eps_1 = alpha * eps_perm

        memoization = True


        hits = 0
        for i in range(n_users):
            if protocol == 'RAPPOR':
                report, user_memo_vector[i], budget_used, hits = RAPPOR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LGRR':
                report, user_memo_vector[i], budget_used, hits = L_GRR_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOSUE':
                report, user_memo_vector[i], budget_used, hits = L_OSUE_Client_TAU(data_encoded[i][0], k, eps_perm, eps_1,user_memo_vector[i], memoization)
            if protocol == 'LOLOHA':
                report, user_memo_vector[i], budget_used, hits = LOLOHA_Client_TAU(data_encoded[i][0], g, eps_perm, eps_1,user_memo_vector[i], memoization)

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

        hit_vector.append(hits)
        tr_vector.append(tr_value)

        # coordenates = xmin, ymin, xmax, ymax 


        # for i in range(len(grid)):
        #     coord_vector.append((grid.iloc[i]['geometry'].bounds))

        quad_tree.set_counts(count_vector)
        quad_tree.calculate_repo_counts()

        # quad_tree.display()

        est_frequency_vector.append(est_fr_vector)     
        
        real_freq = np.nan_to_num(real_freq)
        est_freq =np.nan_to_num(est_freq)
        
        grid_k.append(k) #Save the list of grids used
                                # tr_tau.append(tr)
        lst_mse.append(mean_squared_error(real_freq*10000, est_freq*10000))
        lst_mae.append(mean_absolute_error(real_freq*10000, est_freq*10000))

        if t == 0:
            old_est_freq = est_freq

        similarity_vector.append(np.nan_to_num(cosine(est_freq, old_est_freq)))


        old_est_freq = est_freq

    end_time = time.time()

    running_time = end_time - start_time 

    budget_tracking = np.mean(final_budget_users) 

    results.append((seed,protocol,structure,cell_size,eps_perm,eps_perm_base,bp,quad_size_base,grid_k,budget_tracking,est_frequency_vector,lst_mse,np.mean(lst_mse),lst_mae,np.mean(lst_mae),w,num_points,similarity_vector,running_time, np.mean(hit_vector)))
    

    # return results, grid_coordenates
    return results, []

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
        if s == "AP4NOMERGE":
            results, grid_coordenates = AP4_NOMERGE("LGRR",results,alpha,eps_perm,bp,w,cell_size,s,tr,data,data_info,check_norm,tr_count)
        

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
    if structure == "AP4NOMERGE":
        results, grid_coordenates = AP4_NOMERGE("RAPPOR",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,check_norm,tr_count,notmsub)  


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
    if structure == "AP4NOMERGE":
        results, grid_coordenates = AP4_NOMERGE("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)  
    if structure == "PRIVAG":
        results, grid_coordenates = PrivAG("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)
    if structure == "ALOG":
        results, grid_coordenates = Alog("LOSUE",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)
    

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
    if structure == "AP4NOMERGE":
        results, grid_coordenates = AP4_NOMERGE("LOLOHA",results,seed,eps_perm,bp,w,cell_size,structure,tr,data,data_info,num_points,check_norm,tr_count,notmsub)  


    
    return results, grid_coordenates, data

def losue(seed,eps,cell_size,structure,p,data,data_info,result_dir,num_points,bp,w,count):

    result_dir.mkdir(parents=True, exist_ok=True)

    auxiliar_data = result_dir / "auxiliar_data/"

    auxiliar_data.mkdir(parents=True, exist_ok=True)


    columns_name = ['seed','method','structure','cell_size','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points','similarity','run_time','hits']

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

    columns_name = ['seed','method','structure','cell_size','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points','similarity','run_time','hits']

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

    columns_name = ['seed','method','structure','cell_size','budget','budget_base','budget_proportion','grid_size_base','grid_size','final_budget','est_freq','mse_t','mse_avg','mae_t','mae_avg','w','num_points','similarity','run_time','hits']

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
    print(experiment)
                            
    p = []
    p.append(0.4) # alpha - p[0]
    p.append([float(x) for x in experiment['e']]) # eps_perm - p[1]
    p.append([float(x) for x in experiment['e_prop']]) # budget_proportion - p[2]
    p.append([int(x) for x in experiment['g']]) # cell_size - p[3]
    # p.append(["Uniform","AP2","AP3","AP4","PRIVAG","ALOG"]) # structure - p[4]
    p.append(["Uniform","AP2","AP3","AP4"]) # structure - p[4]
    p.append(10) # nb_seed - p[5]
    p.append([0.01]) # fr - p[6]
    p.append(True) # check_norm - p[7]
    p.append(True) # tr_count - p[8]
    p.append(None) # data_set_type - p[9]
    p.append(None) # data_set_distribution - p[10]
    #p.append(Path().resolve()/ "Dataset/Geolife_Trajectories_Dataset/Taxi/geolife_cartesian_bounded_20_10s.pkl") #p[11]
    p.append(Path().resolve()/ "Dataset/Taxi_Porto_KAGGLE/new_taxi_portugal_10000_20.pkl")
    p.append(experiment['folder'][0]) #p[12]
    p.append([int(x) for x in experiment['w']]) #p[13]
    # p.append(["LOSUE","LOLOHA","RAPPOR"]) #p[14]
    p.append(["LOSUE","LOLOHA","RAPPOR"]) #p[14]
    p.append([int(x) for x in experiment['p']]) #p[15]
    p.append(1000) # Usuarios - p[16]
    p.append(False) # normsub - p[17]
    
    return p    



def generate_data(data_type,num_users,file_path,file_info,syntetic_path):

#############################################################
#                     Generate Data                         #
#############################################################

   
    # to load syntetic data from file
    if data_type == "l":
        #need fix to get information hardcoded in the generation

        with open(file_path, 'rb') as f:
            data = pickle.load(f)
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
                                                                
        data, x_min, y_min, x_max, y_max = gen_execution_data(syntetic_path)

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
   

def process_sample(experiment,result_dir,data, x_min, y_min, x_max, y_max, s):


    ##### PARAMETERS SETUP #####
    
    # results_partial = pd.DataFrame(columns=['budget','budget_proportion','budget_base','cell_size','method','mse','mae','structure','seed','sample','grid_size','num_users','tr'])
    p = get_parameters(experiment)

    p[5] = s

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
                                    for w in p[13]:
                                        if s=='AP2' or 'ALOG':
                                            for bp in p[2]:
                                                tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,bp,w,count)))
                                                count += 1
                                        else:
                                            tasks.append((losue, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,0,1,count)))
                                            count += 1
                            else:
                                if m == "LOLOHA":
                                    if s == "Uniform":
                                        tasks.append((loloha, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,count)))
                                        count += 1
                                    #else:
                                    #    if s == "AP2":
                                    #        for bp in p[2]:
                                    #            for w in p[13]:
                                    #                tasks.append((loloha, (seed,e, c, s, p, new_data, data_info, result_dir,num_points,count)))
                                    #                count += 1
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
        syntetic_path = Path().resolve() / "Dataset/Geolife_Trajectories_Dataset/Taxi/geolife_cartesian_bounded_120.pkl"
    elif data_type == 'p':
        syntetic_path = Path().resolve() / "Dataset/Taxi_Porto_KAGGLE/new_taxi_portugal_10000_120.pkl"
    elif data_type == 'u':
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Uniform/data_uni.pkl"
    elif data_type == 'n':
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Normal/data_norm.pkl"
    else:
        syntetic_path =  Path().resolve() / "Dataset/Sintetic_Direction/data_dir.pkl"

    data, x_min, y_min, x_max, y_max = generate_data(data_type,num_users,file_path,file_info,syntetic_path)

    # print(len(data))
    # print(x_min)
    # print(x_max)
    # print(y_min)
    # print(y_max)
    s = int(experiments[0]['s'][0])

    # Print the parsed experiments
    print("Number of experiments:", int(experiments[0]['n'][0]))
    print("Number of seeds:", int(experiments[0]['s'][0]))
    print("Data Type:", data_type)
    print("Data_Path:",syntetic_path)


    for i in range(2,len(experiments)):
        print(f"Experiment {i-1}: {experiments[i]}")
        print(experiments[i]['folder'][0])
        process_sample(experiments[i],result_dir,data, x_min, y_min, x_max, y_max, s)



    # for setup in range(len(experiments)-1):
    #     process_sample(experiments[setup+1])
    #     process_sample(experiments[2])


if __name__ == "__main__":
    main()

