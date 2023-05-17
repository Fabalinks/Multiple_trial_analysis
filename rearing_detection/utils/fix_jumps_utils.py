import numpy as np
import math
from utils.baseline_method import continuous_detection 
from utils.transform_utils import get_quaternion
from tqdm import tqdm
from scipy.spatial.transform import Rotation 

def calculate_average_sliding_window(x,window = 10):
    former_sum = np.cumsum(x[0:-window])
    latter_sum = np.cumsum(x[window:])+ former_sum[window-1]
    return (latter_sum - former_sum)/window

def couple_flip(diff, flip_slice):
    '''
    '''
    coupled_idx = []
    fir_idx = flip_slice[0]
    single_diff = diff[flip_slice[0]] # uncoupled one
    coupled = False
    for i, idx in enumerate(flip_slice[1:]):
        if coupled == False:
            candicate_diff = diff[idx]
            sec_idx = idx
        else:
            single_diff = diff[idx]
            fir_idx = idx
            coupled = False
            
        if single_diff*candicate_diff <0: # one increase, one decrease
            coupled = True
            candicate_diff = 0 # need to update the candidate one
            
            coupled_idx.append([fir_idx, sec_idx]) # record the coupled idx
        else:
            continue
    return coupled, coupled_idx


def generate_marker_loc(row_id,raw_data):
    '''
    '''
    offset_col = 42
    marker_ids = range(0,4)
    if len(row_id)<=1:
        markers = np.zeros((4,3))
        for i, marker_id in enumerate(marker_ids):
            markers[i,:] =   raw_data.iloc[row_id, int(marker_id*4 + offset_col):int(marker_id*4 + offset_col + 3)].values
    else:
        markers = np.zeros((len(row_id),4,3))
        for i, marker_id in enumerate(marker_ids):
            markers[:,i,:] =   raw_data.iloc[row_id, int(marker_id*4 + offset_col):int(marker_id*4 + offset_col + 3)].values
    
    return markers


def detect_difference(fix_markers, markers):
    all_bool = []
    for i in range(0,3):
        this_bool = markers[:,0,i] == fix_markers[:,0,i]
        all_bool.append(this_bool)

    if (all_bool[0] == all_bool[1]).all() == False:
        raise RuntimeError


def find_max_min(markers, locs = [0,2],idx = 1):
    if np.ndim(markers) ==3:
        n = markers.shape[0]
        min_array = np.zeros((n,3))
        max_array = np.zeros((n,3))
        min_idx = np.argmin((markers[:,locs[0],idx], markers[:,locs[1],idx]),axis=0)
        max_idx = np.argmax((markers[:,locs[0],idx], markers[:,locs[1],idx]),axis=0)
        for i in range(0,n):
            min_array[i,:] = markers[i,locs[min_idx[i]],:]
            max_array[i,:] = markers[i,locs[max_idx[i]],:]
    else:
        min_idx = np.argmin((markers[locs[0],idx], markers[locs[1],idx]),axis=0)
        max_idx = np.argmax((markers[locs[0],idx], markers[locs[1],idx]),axis=0)
        min_array = markers[locs[min_idx],:]
        max_array= markers[locs[max_idx],:]
        
    return min_array, max_array


def detect_jumps(markers,delta_threshold = 0.02,single_threshold = 0.01, delay_tolerance= 2):
    '''
    consider the jump of y axis coor
    '''
    delta_diff = np.diff(markers[:,0,1]) - np.diff(markers[:,2,1]) # marker1 and marker3
    diff_middle = np.diff(markers[:,3,1]) # the diff of the marker in the middle
    
    total_jumps = np.argwhere(np.abs(delta_diff) >= delta_threshold).reshape(-1)
    negative_jumps = np.argwhere((np.abs(np.diff(markers[:,0,1])) <= single_threshold) | (np.abs(np.diff(markers[:,2,1])<single_threshold) )).reshape(-1)
    
    positve_jumps = list(set(total_jumps) - set(negative_jumps))
    
    _, jumps_slice = continuous_detection(np.sort(positve_jumps),discontinu_tolerance=5, total_tolerance=1e10,seperate= True)
    
    return jumps_slice


def detect_transition(markers, diff_thresh = 0.02, discontinu_tolerance = 10,neighbor = 4):
    '''
    '''
    delta_diff = np.diff(markers[:,0,1]) - np.diff(markers[:,2,1]) # marker1 and marker3, difference of the first order differential 
    flip_sign = (markers[0:-1,0,1] - markers[0:-1,2,1]) * (markers[1:,0,1] - markers[1:,2,1]) # the height of marker 1 and marker 3 flip
    delta_height = markers[:,0,1] - markers[:,2,1] # difference betweent the height of two markers
    
    
    flip_loc = np.argwhere(flip_sign <0).reshape(-1)
    non_diff_loc = np.argwhere(np.abs(delta_diff) < diff_thresh).reshape(-1)
    
    full_smooth_loc = np.intersect1d(flip_loc, non_diff_loc) 
    _, smooth_slice = continuous_detection(np.sort(full_smooth_loc), discontinu_tolerance= 10,  total_tolerance=1e10,seperate= True)
    transition_direction = []
    smooth_loc = []

    for i,slice in enumerate(smooth_slice):
        if (markers[slice[0]-1,0,1] > markers[slice[0]-1,2,1]) & (markers[slice[-1]+1,0,1] <= markers[slice[0]+1,2,1]):
            
            if (np.mean(delta_height[slice[-1]+1: slice[-1]+1 + neighbor]) < 0):
                transition_direction.append(1)
                smooth_loc.append(slice[0])
            
        elif (markers[slice[0]-1,0,1] < markers[slice[0]-1,2,1]) & (markers[slice[-1]+1,0,1] >= markers[slice[-1]+1,2,1]):
            if (np.mean(delta_height[slice[-1]+1: slice[-1]+1 + neighbor]) >0):
                transition_direction.append(0)
                smooth_loc.append(slice[0])
            
    return smooth_loc, transition_direction

def detect_jumps_with_distance(markers,jump_slice, marker_quality,smooth_loc, neighbor= 10, inferred_range = 200,):
    fix_markers = np.copy(markers)
    for i,this_slice in enumerate(jump_slice):
        if len(this_slice) <=0:
            continue
        else:
            begin = max(this_slice[0] - neighbor,0)

            end = min(this_slice[-1]+ neighbor, len(markers)-1)
            this_slice = range(begin,end+1)
            referred_begin =  max(this_slice[0] - inferred_range,0)
            referred_end = this_slice[0]
            
            average_quatlity = calculate_average_sliding_window(marker_quality[referred_begin:referred_end,1], window=10)
            good_idx = np.argwhere(average_quatlity>0.8)
            _,good_periods = continuous_detection(good_idx, seperate=True)
            while len(good_periods[0])==0:
                inferred_range+= 10
                referred_begin =  max(this_slice[0] - inferred_range,0)
                referred_end = this_slice[0]
                average_quatlity = calculate_average_sliding_window(marker_quality[referred_begin:referred_end,1], window=10)
                good_idx = np.argwhere(average_quatlity>0.8)
                _,good_periods = continuous_detection(good_idx, seperate=True)
            referred_idx = good_periods[0][-1]
            referred_marker1_height = calculate_average_sliding_window(markers[:,0,1], window = 10)[referred_idx]
            referred_marker3_height = calculate_average_sliding_window(markers[:,2,1],window = 10)[referred_idx]
            
            if referred_marker1_height> referred_marker3_height:
                if smooth_loc not in range(begin,end):
                    min_array, max_array = find_max_min(markers[begin:end])
                    fix_markers[begin:end,0,:] = max_array
                    fix_markers[begin:end,2,:] = min_array
                else:
                    print('transition')
            else:
                if smooth_loc not in range(referred_begin,end):
                    min_array, max_array = find_max_min(markers[begin:end])
                    fix_markers[begin:end,0,:] = min_array
                    fix_markers[begin:end,2,:] = max_array
                else:
                    print('transition')
                
    return fix_markers
            
            
def fix_jumps_with_transition_state(markers, jump_slice,smooth_loc, transition_direction, neighbor = 2):
    '''
    fix flip of markers
    '''
    delta_diff = np.diff(markers[:,0,1]) - np.diff(markers[:,2,1]) # marker1 and marker3
    fix_markers = np.copy(markers)
    
    for i,flip_slice in enumerate(jump_slice):
        if len(flip_slice) <=0:
            continue
        else:
            begin = flip_slice[0]
            end = flip_slice[-1]
            this_slice = flip_slice
            
            closest_transition = np.argsort(np.abs(smooth_loc- np.nanmean(this_slice)))[0] # take the two closest dots
            second_closet =  np.argsort(np.abs(smooth_loc- np.nanmean(this_slice)))[1] 
            # int(closest_transition -1) if np.nanmean(this_slice) < smooth_loc[closest_transition] else int(closest_transition +1)
            adjacent_transtion = [closest_transition, second_closet]
            
            # determine the location of the flip site
            if (transition_direction[min(adjacent_transtion)] == 1) & (transition_direction[max(adjacent_transtion)] == 0):
                stage = 1
            elif (transition_direction[min(adjacent_transtion)] == 0) & (transition_direction[max(adjacent_transtion)] == 1):
                stage = 0
            elif smooth_loc[adjacent_transtion[0]] < np.nanmean(this_slice):
                stage = transition_direction[adjacent_transtion[0]]
             
            elif smooth_loc[adjacent_transtion[0]] > np.nanmean(this_slice):
                stage = np.mod(transition_direction[adjacent_transtion[0]] +1,2)
            else:
                stage = 'transition'
            right_idx = 1 # extension along the right side
            left_idx =1 # extension along the left side
            left_flag = True # if True, continue the fixation of flip on the left side
            right_flag = True
            if stage == 0: # in rearing up state
                min_array, max_array = find_max_min(markers[this_slice])
                fix_markers[this_slice,0,:] = max_array
                fix_markers[this_slice,2,:] = min_array
                
                # to incluce wider range
                while (left_flag == True) & (begin-left_idx> smooth_loc[min(adjacent_transtion)]):
                    if fix_markers[begin - left_idx,0,1] < fix_markers[begin - left_idx,2,1]:
                        min_array, max_array = find_max_min(markers[begin - left_idx], single=True)
                        fix_markers[begin - left_idx,0,:] = max_array
                        fix_markers[begin - left_idx,2,:] = min_array
                        left_idx += 1

                    # to skip the nan gap
                    elif (np.isnan(fix_markers[begin - left_idx,0,1])) | (np.isnan(fix_markers[begin - left_idx,2,1])):
                        left_idx += 1
                        continue
                    else:
                        left_flag = False
                
                while (right_flag == True) & (end+right_idx<smooth_loc[max(adjacent_transtion)]):
                    if fix_markers[end+right_idx,0,1] < fix_markers[end+right_idx,2,1]:
                        min_array, max_array = find_max_min(markers[end+right_idx], single=True)
                        fix_markers[end+right_idx,0,:] = max_array
                        fix_markers[end+right_idx,2,:] = min_array
                        
                        right_idx += 1
                    elif (np.isnan(fix_markers[end+right_idx,0,1])) | np.isnan(fix_markers[end+right_idx,2,1]):
                        right_idx += 1
                        continue
                    
                    else:
                        right_flag = False
                        
            elif stage == 1: # not in rearing state
                min_array, max_array = find_max_min(markers[this_slice])
                fix_markers[this_slice,0,:] = min_array
                fix_markers[this_slice,2,:] = max_array

                 # to incluce wider range
                while (left_flag == True) & (begin-left_idx>=smooth_loc[min(adjacent_transtion)]):
                    if fix_markers[begin - left_idx,0,1] > fix_markers[begin - left_idx,2,1]:
                        min_array, max_array = find_max_min(markers[begin - left_idx], single=True)
                        fix_markers[begin - left_idx,0,:] = min_array
                        fix_markers[begin - left_idx,2,:] = max_array
                        
                        left_idx += 1

                    
                    elif (np.isnan(fix_markers[begin - left_idx,0,1])) | (np.isnan(fix_markers[begin - left_idx,2,1])):
                        left_idx += 1
                        continue
                    
                    else:
                        left_flag = False
                
                while (right_flag == True) & (end+right_idx<smooth_loc[max(adjacent_transtion)]):
                    if fix_markers[end+right_idx,0,1] > fix_markers[end+right_idx,2,1]:
                        min_array, max_array = find_max_min(markers[end+right_idx], single=True)
                        fix_markers[end+right_idx,0,:] = min_array
                        fix_markers[end+right_idx,2,:] = max_array
                        
                        
                        right_idx += 1
                    
                    elif (np.isnan(fix_markers[end+right_idx,0,1])) | np.isnan(fix_markers[end+right_idx,2,1]):
                        right_idx += 1
                        continue
                    
                    else:
                        right_flag = False
            else:
                whether_coupled, coupled_idx = couple_flip(delta_diff, flip_slice)
                if whether_coupled:
                    for this_couple in coupled_idx:
                        begin = max(this_couple[0],0)
                        end = min(this_couple[-1], len(markers)-1)
                        this_slice = range(begin,end+1)
                        min_array, max_array = find_max_min(markers[this_slice])
                
                        if np.nanmean(fix_markers[begin-2:begin,0,1]) > np.nanmean(fix_markers[begin - 2:begin,2,1]):
                            fix_markers[this_slice,0,:] = max_array
                            fix_markers[this_slice,2,:] = min_array
                        else:
                            fix_markers[this_slice,0,:] = min_array
                            fix_markers[this_slice,2,:] = max_array
    return fix_markers


def fix_jumps(markers, jump_slice, state_thresh = 0.6, neighbor = 2):
    '''
    '''
    delta_diff = np.diff(markers[:,0,1]) - np.diff(markers[:,2,1]) # marker1 and marker3
    fix_markers = np.copy(markers)
    for i,this_slice in enumerate(jump_slice):
        if len(this_slice) <=0:
            continue
        else:
            begin = max(this_slice[0] - neighbor,0)
            end = min(this_slice[-1]+ neighbor, len(markers)-1)
            this_slice = range(begin,end+1)
            right_idx = 1
            left_idx =1
            left_flag = True
            right_flag = True
            if max(np.mean(markers[this_slice,0,1]), np.mean(markers[this_slice,2,1])) > state_thresh: # in rearing up state
                min_array, max_array = find_max_min(markers[this_slice])
                fix_markers[this_slice,0,:] = max_array
                fix_markers[this_slice,2,:] = min_array
                       
                # to incluce wider range
                while (left_flag == True) & (begin-left_idx>=0) &(max(fix_markers[begin - left_idx,0,1] ,fix_markers[begin - left_idx,2,1])>state_thresh):
                    if fix_markers[begin - left_idx,0,1] < fix_markers[begin - left_idx,2,1]:
                        min_array, max_array = find_max_min(markers[begin - left_idx])
                        fix_markers[begin - left_idx,0,:] = max_array
                        fix_markers[begin - left_idx,2,:] = min_array
                        left_idx += 1
                        
                    # to skip the nan gap
                    elif (np.isnan(fix_markers[begin - left_idx,0,1])) | (np.isnan(fix_markers[begin - left_idx,2,1])):
                        left_idx += 1
                        continue
                    else:
                        left_flag = False
                
                while (right_flag == True) & (end+right_idx<=len(markers)) &(max(fix_markers[begin - left_idx,0,1] ,fix_markers[begin - left_idx,2,1])>state_thresh):
                    if fix_markers[end+right_idx,0,1] < fix_markers[end+right_idx,2,1]:
                        min_array, max_array = find_max_min(markers[end + right_idx])
                        fix_markers[end + right_idx,0,:] = max_array
                        fix_markers[end + right_idx,2,:] = min_array
                        right_idx += 1
                    elif (np.isnan(fix_markers[end+right_idx,0,1])) | np.isnan(fix_markers[end+right_idx,2,1]):
                        right_idx += 1
                        continue
                    
                    else:
                        right_flag = False
                        
            else: # not in rearing state
                min_array, max_array = find_max_min(markers[this_slice])
                fix_markers[this_slice,0,:] = min_array
                fix_markers[this_slice,2,:] = max_array
                 # to incluce wider range
                while (left_flag == True) & (begin-left_idx>=0) & (min(fix_markers[begin - left_idx,0,1] ,fix_markers[begin - left_idx,2,1])<state_thresh):
                    if fix_markers[begin - left_idx,0,1] > fix_markers[begin - left_idx,2,1]:
                        min_array, max_array = find_max_min(markers[begin - left_idx])
                        fix_markers[begin - left_idx,0,:] = min_array
                        fix_markers[begin - left_idx,2,:] = max_array
                        left_idx += 1
                    
                    elif (np.isnan(fix_markers[begin - left_idx,0,1])) | (np.isnan(fix_markers[begin - left_idx,2,1])):
                        left_idx += 1
                        continue
                    
                    else:
                        left_flag = False
                
                while (right_flag == True) & (end+right_idx<=len(markers)) & (min(fix_markers[begin - left_idx,0,1] ,fix_markers[begin - left_idx,2,1])<state_thresh):
                    if fix_markers[end+right_idx,0,1] > fix_markers[end+right_idx,2,1]:
                        min_array, max_array = find_max_min(markers[end + right_idx])
                        fix_markers[end + right_idx,0,:] = min_array
                        fix_markers[end + right_idx,2,:] = max_array
                        right_idx += 1
                    
                    elif (np.isnan(fix_markers[end+right_idx,0,1])) | np.isnan(fix_markers[end+right_idx,2,1]):
                        right_idx += 1
                        continue
                    
                    else:
                        right_flag = False

            # print(this_slice)
            # print(0)
            # print(fix_markers[this_slice,0,:])
            # print(2)
            # print(fix_markers[this_slice,2,:],)
    
    return fix_markers

def find_flip_index(fix_markers, markers, inclue_nan = False, which_axis = 0):
    '''
    Return the index where the filp of marker happens
    '''
    whether_filp = fix_markers[:,0,0] != markers[:,0,0] # will contain the nan index
    if inclue_nan == True:
        return np.argwhere(whether_filp).reshape(-1)
    else:
        nan_idx = np.isnan(fix_markers[:,-1,0]) # nan index
        return np.argwhere(whether_filp & ~nan_idx).reshape(-1)


def calcualte_angles_after_flip_correction(self_coor, fix_markers, markers, quaternion, inclue_nan = False):
    '''
    '''
    filp_index = find_flip_index(fix_markers, markers, inclue_nan= inclue_nan)
    fixed_quaternion = np.copy(quaternion)
    
    for i,idx in tqdm(enumerate(filp_index)):
        this_translated_marker = fix_markers[idx, :,:] - np.tile(fix_markers[idx,-1,:,],(4,1)) # minusing the translation distance
        if np.sum(np.isnan(this_translated_marker)) != 0:
            r_matrix = np.array([[np.nan, np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]])
            reorder_quant = np.array([np.nan, np.nan, np.nan,np.nan])
        else:
            quant = np.array(get_quaternion(self_coor[0:-1],this_translated_marker[0:-1])) # [qw, qx, qy, qz]
            reorder_quant = quant[[1,2,3,0]] #  [qx,qy,qz,qw]
            r_matrix = Rotation.from_quat(reorder_quant) # rotation matrix 
            euler_angle = r_matrix.as_euler('xzy',degrees=True) # angle calculated from coordinated of marker
        
        ## correct the flip 
        fixed_quaternion[idx] = reorder_quant
    return fixed_quaternion
