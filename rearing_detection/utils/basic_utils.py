import numpy as np


def get_tags(rat_subdirectory):
    '''
    get time_id/tags from the subdirectories of rat
    '''
    tags = []
    for sub in rat_subdirectory:
        
        # eg of sub: '../Data/Raw/FS10/BPositions_FS10_20211025-143051/', so first replace / with _, and split it to get 20211025-143051
        tag = sub.replace('/','_').split('_')[-2]  
        tags.append(tag)
        
    return tags

def calculate_cumulative_distribution(x, step = 0.005, set_max = False, max_x = 1, set_min = False, min_x = -1):
    '''
    x: one-d array 
    set_max, set_min; whether to set the min and max value that you want to take in, this is used when there is exteme value that we do not want to care
    
    '''
    sorted_x = np.sort(x)
    if set_max == False:
        max_x =  sorted_x[-1]
        
    if set_min == False:
        min_x = sorted_x[0]
    
    line_x = np.arange(min_x, max_x, step)
    cumulate_disdribution_x = []
    for i in np.arange(min_x, max_x, step):
        
        cumulate_disdribution_x.append(np.count_nonzero(sorted_x<=i))
    
    return line_x, np.asarray(cumulate_disdribution_x)/len(x) # get the probability