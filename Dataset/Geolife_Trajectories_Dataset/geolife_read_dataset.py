import os
import glob
import pandas as pd

mode_names = ['walk', 'bike', 'bus', 'car', 'subway','train', 'airplane', 'boat', 'run', 'motorcycle', 'taxi']
mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}


def read_plt(plt_file,i,limited=False,len_tr=1):
    points = pd.read_csv(plt_file, skiprows=6, header=None,
                         parse_dates=[[5, 6]], infer_datetime_format=True)

    # for clarity rename columns
    points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    if limited:

        df_filtered = points[(points['time'] - points['time'].shift(1)) >= pd.Timedelta(seconds=60)]

        # A primeira linha sempre será incluída, então adicionamos manualmente
        df_filtered = pd.concat([points.iloc[[0]], df_filtered])

        # print(df_filtered)
        # print("filtered:",len(df_filtered))

        points = df_filtered

    points['tr'] = i
    
    if len(points) > len_tr:
        return points[:len_tr]

    if len(points) < len_tr:
        return None
    

def read_labels(labels_file):
    labels = pd.read_csv(labels_file, skiprows=1, header=None,
                         parse_dates=[[0, 1], [2, 3]],
                         infer_datetime_format=True, delim_whitespace=True)

    # for clarity rename columns
    labels.columns = ['start_time', 'end_time', 'label']

    # replace 'label' column with integer encoding
    labels['label'] = [mode_ids[i] for i in labels['label']]

    return labels

def apply_labels(points, labels):
    indices = labels['start_time'].searchsorted(points['time'], side='right') - 1
    no_label = (indices < 0) | (points['time'].values >= labels['end_time'].iloc[indices].values)
    points['label'] = labels['label'].iloc[indices].values
    points['label'][no_label] = 0

def read_user(user_folder):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    df = pd.concat([read_plt(f,i) for i,f in enumerate(plt_files)],ignore_index=True)


    labels_file = os.path.join(user_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0

    return df
    # return df

def read_few_users(user_folder, num_traj=1, len_tr=1):
    labels = None

    plt_files = glob.glob(os.path.join(user_folder, 'Trajectory', '*.plt'))
    dfs = [] 
    for i, f in enumerate(plt_files):
        if i > num_traj:
            break  # Sai do loop após num_traj iterações
        df_f = read_plt(f, i, True, len_tr)
        if df_f is not None:
            dfs.append(df_f)
        
    if len(dfs) != 0:
        df = pd.concat(dfs, ignore_index=True)

        labels_file = os.path.join(user_folder, 'labels.txt')
        if os.path.exists(labels_file):
            labels = read_labels(labels_file)
            apply_labels(df, labels)
        else:
            df['label'] = 0

        return df
    else:
        return None

def read_all_users(folder):
    subfolders = os.listdir(folder)
    # subfolders = ['107']
    dfs = []
    
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i, len(subfolders)-1, sf))
        df = read_user(os.path.join(folder,sf))
        df['user'] = int(sf)
        dfs.append(df)
    
    return pd.concat(dfs,ignore_index=True)

def generate_pickle_for_each_user(folder):
    subfolders = os.listdir(folder)
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing user %s' % (i, len(subfolders)-1, sf))

        #labels = None

        #plt_files = glob.glob(os.path.join(folder, 'Trajectory', '*.plt'))

        
        subfolders = os.listdir(folder)
        # subfolders = ['107']
        dfs = []
        
        # for i, sf in enumerate(subfolders):
        #     print('[%d/%d] processing user %s' % (i, len(subfolders)-1, sf))
        #     read_user_by_demand(os.path.join(folder,sf))
        
        # df = read_user(os.path.join(folder,sf))
        
    
def read_sample_user(folder,num_users=10,num_tr=10,len_tr=10):
    subfolders = os.listdir(folder)
    # subfolders = ['107']
    dfs = []
    for i, sf in enumerate(subfolders):
        if i>num_users:
            break
        print('[%d/%d] processing user %s' % (i, len(subfolders)-1, sf))
        df = read_few_users(os.path.join(folder,sf),num_tr,len_tr)
        if df is not None:
            df['user'] = int(sf)
            dfs.append(df)
        
    
    return pd.concat(dfs,ignore_index=True)

def generate_pickle_file(df:pd.DataFrame, path:str):
    df.to_pickle(path)

def read_pickle(path:str):
    return pd.read_pickle(path)


