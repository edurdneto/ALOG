import numpy as np
from numba import jit
import xxhash
from sys import maxsize

# [1] Erlingsson, Pihur, and Korolova (2014) "RAPPOR: Randomized aggregatable privacy-preserving ordinal response" (ACM CCS).
# [2] Arcolezi et al (2022) "Improving the Utility of Locally Differentially Private Protocols for Longitudinal and Multidimensional Frequency Estimates" (Digital Communications and Networks).
# [3] Ding, Kulkarni, and Yekhanin (2017) "Collecting telemetry data privately." (NeurIPS).

normsub_v1 = True

@jit(nopython=True)
def setting_seed(seed):
    """ Function to set seed for reproducibility.
    Calling numpy.random.seed() from interpreted code will 
    seed the NumPy random generator, not the Numba random generator.
    Check: https://numba.readthedocs.io/en/stable/reference/numpysupported.html"""
    
    np.random.seed(seed)

@jit(nopython=True)
def GRR_Client(input_data, k, p):
    """
    Generalized Randomized Response (GRR) protocol
    """
    domain = np.arange(k) 

    # GRR perturbation function
    rnd = np.random.random()
    if rnd <= p:
        return input_data

    else:
        return np.random.choice(domain[domain != input_data])

@jit(nopython=True)
def UE_Client(input_ue_data, k, p, q):
    
    """
    Unary Encoding (UE) protocol
    """
    
    # Initializing a zero-vector
    sanitized_vec = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                sanitized_vec[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                sanitized_vec[ind] = 1
    return sanitized_vec

def norm_sub(est_freq, p, q):

    n_total = np.sum(est_freq)

    noise_expected = n_total * q
    est_freq_adjusted = est_freq - noise_expected

    est_freq_adjusted = est_freq_adjusted / (p - q)

    est_freq_adjusted = np.clip(est_freq_adjusted, 0, None)

    est_freq_adjusted = est_freq_adjusted / np.sum(est_freq_adjusted) * n_total

    return est_freq_adjusted

def LOLOHA_Client_TAU(ts_value, g, eps_perm, eps_1,memo_vector, memoization):
    
    # GRR parameters for round 1
    p1_llh = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
    q1_llh = (1 - p1_llh) / (g-1)
    
    # GRR parameters for round 2
    p2_llh = (q1_llh - np.exp(eps_1) * p1_llh) / ((-p1_llh * np.exp(eps_1)) + g*q1_llh*np.exp(eps_1) - q1_llh*np.exp(eps_1) - p1_llh*(g-1)+q1_llh)
    q2_llh = (1 - p2_llh) / (g-1)
    
    
    # Random 'hash function', i.e., a seed to use xxhash package
    rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
    # Hash the user's value
    hashed_input_data = (xxhash.xxh32(str(ts_value), seed=rnd_seed).intdigest() % g)

    if memoization:
        if memo_vector[hashed_input_data] is None: # If hashed value not memoized
            # Memoization
            first_sanitization = GRR_Client(hashed_input_data, g, p1_llh)
            memo_vector[hashed_input_data] = first_sanitization
            budget_used = eps_perm
        
        else: # Use already memoized hashed value
            first_sanitization = memo_vector[hashed_input_data]
            budget_used = 0
    else:
        first_sanitization = GRR_Client(hashed_input_data, g, p1_llh)
        budget_used = eps_perm

    sanitized_report = (GRR_Client(first_sanitization, g, p2_llh), rnd_seed)
    
    return sanitized_report, memo_vector, budget_used

def LOLOHA_Aggregator_TAU(reports, k, eps_perm, eps_1, g,nsub=True):    
        
    # Number of reports
    n = len(reports)
    
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + g - 1)
    q1 = (1 - p1) / (g-1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + g*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(g-1)+q1)
    q2 = (1 - p2) / (g-1)
    
    # Count how many times each value has been reported
    q1 = 1 / g #updating q1 in the server        
    count_report = np.zeros(k)
    for tuple_val_seed in reports:
        usr_val = tuple_val_seed[0] # user's sanitized value
        usr_seed = tuple_val_seed[1] # user's 'hash function'
        for v in range(k):
            if usr_val == (xxhash.xxh32(str(v), seed=usr_seed).intdigest() % g):
                count_report[v] += 1
    

    #FOR NORM SUB
    if nsub:
        est_freq = ((count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2)))
        est_freq = norm_sub(est_freq,p2,q2)
        est_freq = np.nan_to_num(est_freq)
    
    else:
        # Ensure non-negativity of estimated frequency
        est_freq = ((count_report - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2)))
       
        est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return est_freq


def RAPPOR_Client_TAU(ts_value, k, eps_perm, eps_1,memo_vector, memoization):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]
    
    # Sue parameters for round 1
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters for round 2
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                        np.exp(eps_perm) - 1) ** 2 - (
                        np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                    eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                        np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                    eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                    eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
                np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
            eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    
    # Unary encoding
    input_ue_data = np.zeros(k)
    input_ue_data[ts_value] = 1

    if memoization:    
        if memo_vector[ts_value] is None: # If hashed value not memoized

            # Memoization
            first_sanitization = UE_Client(input_ue_data, k, p1, q1)
            memo_vector[ts_value] = first_sanitization
            budget_used = eps_perm

        else: # Use already memoized hashed value
            first_sanitization = memo_vector[ts_value]
            budget_used = 0
    else:
        first_sanitization = UE_Client(input_ue_data, k, p1, q1)
        budget_used = eps_perm
    
    sanitized_report = UE_Client(first_sanitization, k, p2, q2)
    
    return sanitized_report, memo_vector, budget_used

def RAPPOR_Aggregator(ue_reports, eps_perm, eps_1,nsub=True):

    # Number of reports
    n = len(ue_reports)

    # SUE parameters for round 1
    p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
    q1 = 1 - p1

    # SUE parameters for round 2
    p2 = - (np.sqrt((4 * np.exp(7 * eps_perm / 2) - 4 * np.exp(5 * eps_perm / 2) - 4 * np.exp(
        3 * eps_perm / 2) + 4 * np.exp(eps_perm / 2) + np.exp(4 * eps_perm) + 4 * np.exp(3 * eps_perm) - 10 * np.exp(
        2 * eps_perm) + 4 * np.exp(eps_perm) + 1) * np.exp(eps_1)) * (np.exp(eps_1) - 1) * (
                        np.exp(eps_perm) - 1) ** 2 - (
                        np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(
                    eps_1 + eps_perm) + np.exp(eps_1 + 2 * eps_perm) - 1) * (
                        np.exp(3 * eps_perm / 2) - np.exp(eps_perm / 2) + np.exp(eps_perm) - np.exp(
                    eps_1 + eps_perm / 2) - np.exp(eps_1 + eps_perm) + np.exp(eps_1 + 3 * eps_perm / 2) + np.exp(
                    eps_1 + 2 * eps_perm) - 1)) / ((np.exp(eps_1) - 1) * (np.exp(eps_perm) - 1) ** 2 * (
                np.exp(eps_1) - np.exp(2 * eps_perm) + 2 * np.exp(eps_perm) - 2 * np.exp(eps_1 + eps_perm) + np.exp(
            eps_1 + 2 * eps_perm) - 1))
    q2 = 1 - p2

    #FOR NORM SUB
    if nsub:
        est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2)))
        est_freq = norm_sub(est_freq,p2,q2)
        est_freq = np.nan_to_num(est_freq)

    else:
        # Ensure non-negativity of estimated frequency
        est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

        est_freq = np.nan_to_num(est_freq/sum(est_freq))
        
    return est_freq


def L_OSUE_Client_TAU(ts_value, k, eps_perm, eps_1,memo_vector, memoization):
    
    # OUE parameters for round 1
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters for round 2
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    
    # Unary encoding
    input_ue_data = np.zeros(k)
    input_ue_data[ts_value] = 1
    if memoization:
        if memo_vector[ts_value] is None: # If hashed value not memoized

            # Memoization
            first_sanitization = UE_Client(input_ue_data, k, p1, q1)
            memo_vector[ts_value] = first_sanitization
            budget_used = eps_perm

        else: # Use already memoized hashed value
            first_sanitization = memo_vector[ts_value]
            budget_used = 0
    else:
        first_sanitization = UE_Client(input_ue_data, k, p1, q1)
        budget_used = eps_perm
        
    sanitized_report = UE_Client(first_sanitization, k, p2, q2)
    
    
    return sanitized_report, memo_vector, budget_used

def L_OSUE_Aggregator(ue_reports, eps_perm, eps_1,nsub=True):

    # Number of reports
    n = len(ue_reports)

    # OUE parameters for round 1
    p1 = 1 / 2
    q1 = 1 / (np.exp(eps_perm) + 1)

    # SUE parameters for round 2
    p2 = (1 - np.exp(eps_1 + eps_perm)) / (np.exp(eps_1) - np.exp(eps_perm) - np.exp(eps_1 + eps_perm) + 1)
    q2 = 1 - p2

    #FOR NORM SUB
    if nsub:
        est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2)))
        est_freq = norm_sub(est_freq,p2,q2)
        est_freq = np.nan_to_num(est_freq)

    else:
    
        # Ensure non-negativity of estimated frequency
        est_freq = ((sum(ue_reports) - n * q1 * (p2 - q2) - n * q2) / (n * (p1 - q1) * (p2 - q2))).clip(0)

        est_freq = np.nan_to_num(est_freq / sum(est_freq))
        
    return est_freq

def L_GRR_Client_TAU(ts_value, k, eps_perm, eps_1, memo_vector, memoization):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]
    
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (np.exp(eps_1+eps_perm) - 1) / (-k*np.exp(eps_1)+(k-1)*np.exp(eps_perm)+np.exp(eps_1)+np.exp(eps_1+eps_perm)-1)
    
    q2 = (1 - p2) / (k-1)
    
    if memoization:    
        if memo_vector[ts_value] is None: # If hashed value not memoized
        
            # Memoization
            first_sanitization = GRR_Client(ts_value, k, p1)
            memo_vector[ts_value] = first_sanitization
            budget_used = eps_perm

        else: # Use already memoized hashed value
            first_sanitization = memo_vector[ts_value]
            budget_used = 0
        
    else:
        first_sanitization = GRR_Client(ts_value, k, p1)
        budget_used = eps_perm

    sanitized_report =GRR_Client(first_sanitization, k, p2)

    
    return sanitized_report, memo_vector, budget_used

# Competitor: L-GRR [2]
def L_GRR_Client(input_sequence, k, eps_perm, eps_1):
    # The analytical analysis of how to calculate parameters (p1, q2, p2, q2) is from [2]
    
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (np.exp(eps_1+eps_perm) - 1) / (-k*np.exp(eps_1)+(k-1)*np.exp(eps_perm)+np.exp(eps_1)+np.exp(eps_1+eps_perm)-1)
    
    q2 = (1 - p2) / (k-1)
    
    # Cache for memoized values
    lst_memoized = {val:None for val in range(k)}
    
    # List of sanitized reports throughout \tau data collections
    sanitized_reports = []
    for input_data in input_sequence:
        
        if lst_memoized[input_data] is None: # If hashed value not memoized
        
            # Memoization
            first_sanitization = GRR_Client(input_data, k, p1)
            lst_memoized[input_data] = first_sanitization
        
        else: # Use already memoized hashed value
            first_sanitization = lst_memoized[input_data]
        
        sanitized_reports.append(GRR_Client(first_sanitization, k, p2))

        #print("p1:",p1,"q1:",q1,"p2:",p2,"q2",q2,"input_data:",input_data,"first_sanitization:",first_sanitization)
    # Number of data value changes, i.e, of privacy budget consumption
    final_budget = sum([val is not None for val in lst_memoized.values()])
    
    return sanitized_reports, final_budget

def L_GRR_Aggregator(reports, k, eps_perm, eps_1):

    # Number of reports
    n = len(reports)
                
    # GRR parameters for round 1
    p1 = np.exp(eps_perm) / (np.exp(eps_perm) + k - 1)
    q1 = (1 - p1) / (k - 1)

    # GRR parameters for round 2
    p2 = (q1 - np.exp(eps_1) * p1) / ((-p1 * np.exp(eps_1)) + k*q1*np.exp(eps_1) - q1*np.exp(eps_1) - p1*(k-1)+q1)
    q2 = (1 - p2) / (k-1)

    # Count how many times each value has been reported
    count_report = np.zeros(k)            
    for rep in reports:
        count_report[rep] += 1

    # Ensure non-negativity of estimated frequency
    est_freq = ((count_report - n*q1*(p2-q2) - n*q2) / (n*(p1-q1)*(p2-q2))).clip(0)

    # Re-normalized estimated frequency
    est_freq = np.nan_to_num(est_freq / sum(est_freq))

    return est_freq

# Competitor: dBitFlipPM [3]
def dBitFlipPM_Client(input_sequence, k, b, d, eps_perm):
    
    def dBit(bucketized_data, b, d, j, eps_perm):
        
        # SUE parameters
        p1 = np.exp(eps_perm / 2) / (np.exp(eps_perm / 2) + 1)
        q1 = 1 - p1
        
        # Unary encoding
        permanent_sanitization = np.ones(b) * - 1 # set to -1 non-sampled bits

        # Permanent Memoization
        idx_j = 0
        for i in range(b):
            if i in j: # only the sampled bits
                rand = np.random.random()
                if bucketized_data == j[idx_j]:
                    permanent_sanitization[j[idx_j]] = int(rand <= p1)
                else:
                    permanent_sanitization[j[idx_j]] = int(rand <= q1)

                idx_j+=1
        return permanent_sanitization
    
    # calculate bulk number of user's value
    bulk_size = k / b
    
    # bucketized sequence
    bucket_sequence = [int(input_data / bulk_size) for input_data in input_sequence]
    
    # Select random bits and permanently memoize them
    j = np.random.choice(range(0, b), d, replace=False)
    
    # UE matrix of b buckets
    UE_b = np.eye(b)
    
    # mapping {0, 1}^d possibilities of input data
    mapping_d = np.unique([UE_b[val][j] for val in bucket_sequence], axis=0)
    
    # Privacy budget consumption min(d+1, b)
    final_budget = len(mapping_d)
        
    # Cache for memoized values
    lst_memoized = {str(val): None for val in mapping_d}
    
    # List of sanitized reports throughout \tau data collections 
    sanitized_reports = []
    for bucketized_data in bucket_sequence:
        
        pattern = str(UE_b[bucketized_data][j])
        if lst_memoized[pattern] is None: # Memoize value
        
            first_sanitization = dBit(bucketized_data, b, d, j, eps_perm)
            lst_memoized[pattern] = first_sanitization
        
        else: # Use already memoized value
            first_sanitization = lst_memoized[pattern]
        
        sanitized_reports.append(first_sanitization)

    # Number of memoized responses
    nb_changes = len(np.unique([val for val in lst_memoized.values() if val is not None], axis = 0))
    
    # Boolean value to indicate if number of memoized responses equal number of bucket value changes
    detect_change = len(np.unique(bucket_sequence)) == nb_changes

    return sanitized_reports, final_budget, detect_change

def dBitFlipPM_Aggregator(reports, b, d, eps_perm):
    
    # Estimated frequency of each bucket
    est_freq = []
    for v in range(b):
        h = 0
        for bi in reports:
            if bi[v] >= 0: # only the sampled bits
                h += (bi[v] * (np.exp(eps_perm / 2) + 1) - 1) / (np.exp(eps_perm / 2) - 1)
        est_freq.append(h * b / (len(reports) * d ))
    
    # Ensure non-negativity of estimated frequency
    est_freq = np.array(est_freq).clip(0)
    
    # Re-normalized estimated frequency
    norm_est = np.nan_to_num(est_freq / sum(est_freq))
    return norm_est