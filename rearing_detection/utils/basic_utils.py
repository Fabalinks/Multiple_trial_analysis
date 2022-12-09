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

def density_function(x, step = 0.005,set_max = False, max_x = 1, set_min = False, min_x = -1):
    '''
    calculate the density of given x
    x: array
    step: step to calculate functions 
    '''
    sorted_x = np.sort(x)
    
    if set_max == False:
        max_x =  sorted_x[-1]
        
    if set_min == False:
        min_x = sorted_x[0]
    
    line_x = np.arange(min_x, max_x, step)
    density = []
    for i in np.arange(min_x, max_x, step):
        
        density.append(np.count_nonzero((i<=sorted_x) & (sorted_x<i+step)))
    
    return line_x, np.asarray(density)/len(x) # get the probability

def joint_density_function(x,y, steps = [0.005,0.005],set_max = False, max_x = [1,1], set_min = False, min_x = [-1,-1]):
    '''
    calculate the joint density of x and y
    x , y: array, x and y has the same shape
    steps: step to calculate functions, step[0] for x, steps[1] for y
    max_x: if set_max is True, provide max_x in advance, max_z[0] for x, max_x[1] for y
    '''
    all_input = []
    all_input.append(np.asarray(x).reshape(-1))
    all_input.append(np.asarray(y).reshape(-1)) 

    
    sorted_input = np.sort(all_input) # return [2, n]
    if set_max == False:
        max_x =  sorted_input[:, -1]
        
    if set_min == False:
        min_x = sorted_input[:, 0] # min_x[0] for x, 1 for y
        
    fir_range = np.arange(min_x[0], max_x[0]+ steps[0], steps[0])
    sec_range = np.arange(min_x[1], max_x[1]+ steps[1] , steps[1])
    
    joint_array = np.zeros((len(fir_range), len(sec_range)))
    
    for i, fir_value in enumerate(fir_range):
        fir_idx = np.argwhere((fir_value<=all_input[0]) & (all_input[0]< fir_value + steps[0] )).reshape(-1) # idx that in the range of first feature
        
        
        for j, sec_value in enumerate(sec_range):
            sec_idx = np.argwhere((sec_value <= all_input[1][fir_idx])  & (all_input[1][fir_idx] < sec_value + steps[1])).reshape(-1) # in the range of fir and sec feature

            joint_array[i,j] = len(sec_idx) # count 
        
    
    return [fir_range, sec_range], np.round(joint_array/len(x.reshape(-1)), 20).T# get the probability
