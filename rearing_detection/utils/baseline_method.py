import numpy as np


def detect_whether_rearing(height, z_speed, height_thresh = 0.65, zspeed_thresh = 0.5):
    '''
    Parameters:
        features: rats motion data in a series n bins,
                [n*4] np.array, [:,0]:z, [:,1]: z-speed, [:,2]: pitch, [:,3]:xy-speed 
        height_thresh: if exist z >= height_thresh, then there is rearing pogesture in the n bins
        zspeed_thresh: if exist z-speed >= height_thresh or z-speed <= -height_thresh, then there is rearing pogesture in the n bins
    
    return:
        True: there exist rearing in n bins
        False: there does not exist rearing in n bins
    '''
    
    # determine by height
    over_bins = np.count_nonzero(height>= height_thresh)
    
    if over_bins>0:
        return True
    else:
        positive_bins = np.count_nonzero(z_speed>= zspeed_thresh)
        negative_bins = np.count_nonzero(z_speed <= -zspeed_thresh)
        sum_bins = positive_bins + negative_bins
        
        if sum_bins > 0:
            return True
        else:
            return False




def continuous_detection(index_array, discontinu_tolerance = 2, total_tolerance = 4, seperate = False):
    '''
    Parameter:
        index_array: array of index, [1*n]
        discontinu_tolerance: tolerate the gap between two adjacent one
        total_tolerance: if total number of the discontinous > total_tolerance, return False
        seperate: when seperate == True, if not continuous, slice the index_array into continous parts
    return:
        if seperate is True:
            flag: bool for whether continuous
        if seperate is False:
            flag: bool for whether continuous
            index_slice: continous slice
    '''
    
    left_shift = index_array -1
    
    non_continue_bool = index_array[0:-1] != left_shift[1:] # if idx = 6 is True, it means there is a gap between  index_array[6] and index_array[7] 
    non_continue_idx = np.argwhere(non_continue_bool==True).flatten()        

    flag = True # whether continuous
    
    if seperate == False:
        
        if np.count_nonzero(non_continue_bool) ==0:
            return flag

        elif np.count_nonzero(non_continue_bool) > total_tolerance:
            flag = False
        else:
            for idx in non_continue_idx:
                if np.abs(index_array[idx] - index_array[idx+1]) > discontinu_tolerance:
                    flag = False
                else:
                    continue
            return flag
    
    
    else: # need to seperate         
        last_end = 0
        index_slice = []
        if np.count_nonzero(non_continue_bool) ==0: # true continuous
            return flag, index_slice
        
        else:
            for idx in non_continue_idx:
                if np.abs(index_array[idx] - index_array[idx+1]) > discontinu_tolerance:
                    index_slice.append(index_array[last_end:idx])
                    last_end = idx+1
                    flag = False
                else:
                    continue
                
                # to include the final slice
            if last_end < len(index_array):
                index_slice.append(index_array[last_end:])
                
            
            if np.count_nonzero(non_continue_bool) > total_tolerance:
                flag = False
            
        return flag, index_slice


def extend_down_period(high_period, features, xspeed_drop_thresh = 0.3, total_tolerance = 3):
    '''
    rearing period already include the high height period, try to extend the right of this period to include more falling/down period
    
    Parameters:
        high_period: list of index of bins where z/height >= height threshold
        
        features: rats motion data in a series n bins,
                    [n*4] np.array, [:,0]:z, [:,1]: z-speed, [:,2]: pitch, [:,3]:xy-speed 
        xspeed_drop_thresh: if xy_speed[i] > xspeed_drop_thresh, we do not treat this point as rearing, will stop extend the high-height period
        
        total_tolerance: number of the bin with positive z-speed for falling period that we can tolerate, if # > total_tolerance, stop extending
    
    Return:
        nega_period: list of down period index

    '''
    n = features.shape[0]
    nega_period = []

    ## extend from high height period
    # check whether we extend the rearing period to another side of max-speed idx
    gap = 0
    idx_h = high_period[-1]+1 # the left side of down period
    while (idx_h < n) & (gap < total_tolerance):
        if (features[idx_h,1] <= 0) & (features[idx_h, 2] <= xspeed_drop_thresh):
            nega_period.append(idx_h)
            idx_h += 1
        else:
            gap += 1
            nega_period.append(idx_h)
            idx_h += 1
        
        # posi_period = posi_period[0:-2] # drop the last two
    
    return nega_period



def extend_up_period(high_period, features, xspeed_drop_thresh = 0.3, total_tolerance = 3):
    '''
    rearing period already include the high height period, try to extend the left of this period to include more rising/up period
    Parameters:
        high_period: list of index of bins where z/height >= height threshold
        
        features: rats motion data in a series n bins,
                    [n*4] np.array, [:,0]:z, [:,1]: z-speed, [:,2]: pitch, [:,3]:xy-speed 
        xspeed_drop_thresh: if xy_speed[i] > xspeed_drop_thresh, we do not treat this point as rearing, will stop extend the high-height period
        
        total_tolerance: number of the bin with negative z-speed for rising period that we can tolerate, if # > total_tolerance, stop extending
    
    return:
        posi_period: list of up period index
    
    '''

    posi_period = []
    # extend from high height
    idx_h = high_period[0] -1
    
    gap = 0
    while (idx_h >=0) & (gap < total_tolerance) :
        if (features[idx_h,1] >= 0) & (features[idx_h, 2] <= xspeed_drop_thresh):
            posi_period.append(idx_h)
            idx_h -= 1
        else:
            gap += 1
            posi_period.append(idx_h)
            idx_h -= 1
        # posi_period = posi_period[0:-2] # drop the last two
    
    return posi_period



def extend_highest_speed(max_idx,features, label = 'up', total_tolerance= 2, zspeed_thresh = 0.5, xspeed_drop_thresh= 0.3 ):
    '''
    when no high-height period, there might exist up-period or down-period(incomplete rearing), we detect it find a bin with high z-speed, and also 
    a relatively high hight, then extend this period 
    
    Parameters:
        max_id: int, highest z-speed index
        features: rats motion data in a series n bins,
                        [n*4] np.array, [:,0]:z, [:,1]: z-speed, [:,2]: pitch, [:,3]:xy-speed 
        label: which period to detect, up(rising) or down(falling)
        total_tolerance: number of the bin with negative z-speed for rising period that we can tolerate, if # > total_tolerance, stop extending, vise versa for up peirod
        zspeed_thresh: when z_speed[max_idx] >= zpeed_threh, start to treat it as rearing 
        xspeed_drop_thresh: if xy_speed[i] > xspeed_drop_thresh, we do not treat this point as rearing, will stop extend the high-height period
    
    return:
        period: list of rearing index

    '''
    n = features.shape[0]
    if label == 'up':
        if features[max_idx,1] < zspeed_thresh:
            return []
    else:
        if features[max_idx,1] > -zspeed_thresh:
            return []
    
    
    # for left side
    idx_1 = max_idx -1
    gap = 0
    period = []
    
    period.append(max_idx)
    
    if label == 'up':
        while (idx_1 >=0) & (gap < total_tolerance) :
            if (features[idx_1,1] >= 0) & (features[idx_1, 2] <= xspeed_drop_thresh):
                period.append(idx_1)
                idx_1 -= 1
            else:
                gap += 1
                period.append(idx_1)
                idx_1 -= 1
    else:
        while (idx_1 >=0) & (gap < total_tolerance) :
            if (features[idx_1,1] <= 0) & (features[idx_1, 2] <= xspeed_drop_thresh):
                period.append(idx_1)
                idx_1 -= 1
            else:
                gap += 1
                period.append(idx_1)
                idx_1 -= 1
    
    ## for right side

    idx_2 = max_idx +1
    gap = 0
        
    if label == 'up':
        while (idx_2 < n) & (gap < total_tolerance):
            if (features[idx_2,1] >= 0) & (features[idx_2, 2] <= xspeed_drop_thresh):
                period.append(idx_2)
                idx_2 += 1
            else:
                gap += 1
                period.append(idx_2)
                idx_2 += 1
        

    else:
        while (idx_2 < n) & (gap < total_tolerance):
            if (features[idx_2,1] <= 0) & (features[idx_2, 2] <= xspeed_drop_thresh):
                period.append(idx_2)
                idx_2 += 1
            else:
                gap += 1
                period.append(idx_2)
                idx_2 += 1
    return period
                





def determine_rearing_period(features, height_thresh = 0.65, zspeed_thresh = 0.5, xspeed_drop_thresh = 0.3, gap_tolerance= 1, total_tolerance = 3, height_subthresh = 0.55):
    '''
    detect the rearing period 
    
    Parameters:
        features: rats motion data in a series n bins,
                    [n*3] np.array, [:,0]:z, [:,1]: z-speed, [:,2]: xy-speed; [:,3]: pitch
        
        height_thresh: if z/height[i]>= height_thresh, take it into rearing peirod, we call it high-height period
        
        zspeed_thresh: when z_speed[max_idx] >= zpeed_threh, we will evaluate whether it's rearing 
        
        xspeed_drop_thresh: if xy_speed[i] > xspeed_drop_thresh, we do not treat this point as rearing, will stop extend from exist rearing period
        
        gap_tolerance: tolerate the gap between two adjacent one
        
        total_tolerance: number of the bin with that do not satisfy requirments to be rearing,  if # > total_tolerance, stop extending
        
        height_subthresh: when no high-height period, there might exist up-period or down-period(incomplete rearing), we detect it find a bin with high z-speed, and also 
        a relatively high hight(z >= height_subthresh), then extend this period 
    

    return:
        rearing_period: array of index of rearing bins
    '''
    n = features.shape[0]
    max_posi_idx = np.argmax(features[:,1])
    max_nega_idx = np.argmax(-features[:,1])
    
    ## whether rearing:
    whether_rearing = detect_whether_rearing(height= features[:,0], z_speed=features[:,1], height_thresh= height_thresh, zspeed_thresh= zspeed_thresh,)
    if whether_rearing == False:
        return [] # there is no rearing time
    
    else: # there is rearing time
    
        #### include period with a high height (>= height_thresh)
        high_period = np.argwhere(features[:,0]>= height_thresh).flatten()
        
        if len(high_period)>0: # high_period: # equal to if len(high_period)>0
            
            whether_continu, index_slice= continuous_detection(high_period, discontinu_tolerance=20, total_tolerance= 5, seperate=True)
            
            if (whether_continu == False) & (len(index_slice)!=0): # if len(index_slice) == 0, means no need to seperate
                # print('more than one period of high height period')
                total_high_periods = []
                total_nega_periods = []
                total_posi_periods = []
                for this_slice in index_slice:
                    
                    ## check each slice
                    posi_period = []
                    max_posi_idx = np.argmax(features[:,1])
                    if features[max_posi_idx,1] >= zspeed_thresh:
                        posi_period = extend_up_period(this_slice, max_posi_idx, features, xspeed_drop_thresh= xspeed_drop_thresh, gap_tolerance=gap_tolerance, total_tolerance = total_tolerance)
                    else:
                        posi_period = []
                        
                    ## incluce the down state
                    nega_period = []
                    max_nega_idx = np.argmax(-features[:,1])
                    if features[max_nega_idx,1] <= -zspeed_thresh:
                        nega_period = extend_down_period(this_slice, max_nega_idx, features, xspeed_drop_thresh=xspeed_drop_thresh, gap_tolerance=gap_tolerance, total_tolerance = total_tolerance)
                    else:
                        nega_period = []
                    
                    
                    ## add the current period to the total high_periods
                    total_posi_periods.extend(posi_period)
                    total_nega_periods.extend(nega_period)
                    total_high_periods.extend(this_slice.tolist())
                    
                
            
            
            else: # when there is only one period of high-height period
                # incluce the up state
                total_posi_periods = []
                max_posi_idx = np.argmax(features[:,1])
                if features[max_posi_idx,1] >= zspeed_thresh:
                    total_posi_periods = extend_up_period(high_period, features, xspeed_drop_thresh= xspeed_drop_thresh, total_tolerance = total_tolerance)
                else:
                    total_posi_periods = []
                    
                
                ## incluce the down state
                total_nega_periods = []
                max_nega_idx = np.argmax(-features[:,1])
                if features[max_nega_idx,1] <= -zspeed_thresh:
                    total_nega_periods = extend_down_period(high_period, features, xspeed_drop_thresh=xspeed_drop_thresh, total_tolerance = total_tolerance)
                else:
                    total_nega_periods = []
                    
                
                # update the total high period 
                total_high_periods = high_period.tolist()
                
        
        if len(high_period) == 0: # only high speed(up or down period)
            
            # for up period
            max_posi_idx = np.argmax(features[:,1])
            if features[max_posi_idx,0]>= height_subthresh:
                total_posi_periods = extend_highest_speed(max_posi_idx, features, label='up', total_tolerance= total_tolerance, zspeed_thresh=zspeed_thresh, xspeed_drop_thresh=xspeed_drop_thresh,) 
            else:
                total_posi_periods = []
            ## for down speed
            max_nega_idx = np.argmax(-features[:,1])
            if features[max_nega_idx,0]>= height_subthresh:
                total_nega_periods = extend_highest_speed(max_nega_idx, features, label='down', total_tolerance= total_tolerance, zspeed_thresh=zspeed_thresh, xspeed_drop_thresh=xspeed_drop_thresh,) 
            else:
                total_nega_periods =[]
            
            # update the total high period 
            total_high_periods = []
            
        
    
        rearing_period = []
        rearing_period.extend(total_high_periods)
        rearing_period.extend(total_posi_periods)
        rearing_period.extend(total_nega_periods)
        
        return np.unique(np.sort(rearing_period))
    
    
    
    