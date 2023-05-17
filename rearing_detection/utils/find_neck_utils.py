import numpy as np

def finding_neck(rigid_body_basis, center_marker, use_index):
    '''
    finding neck position relative to the center marker, use Justin's method
    '''
    
    shift_range = np.arange(-0.1,0.1, 0.001) # in meter unit
    grad = np.zeros((3, len(shift_range),3))
    
    for dim in range(0,3):
        offset = np.zeros(3)
        offset[dim] = 1
        offset_basis = rigid_body_basis @ offset
        
        for i, this_shift in enumerate(shift_range):
            shift_center = center_marker + this_shift* offset_basis
            differentail_center = (np.roll(shift_center, -1, axis=0) - np.roll(shift_center,1, axis = 0)) #this_session.sample_rate
            # squre_diff = np.matmul(differentail_center.reshape((differentail_center.shape[0],1,3)), rigid_body_basis)[use_index].reshape((squre_diff.shape[0],3)) 
            squre_diff = np.matmul(rigid_body_basis, differentail_center.reshape(differentail_center.shape[0],3,1))[use_index]
            right_range_diff = remove_outlier(squre_diff.reshape((squre_diff.shape[0],3)), 0.1) # take the right range of speed
            velocity_center = np.sqrt(np.nansum(np.square(right_range_diff), axis = 0)) # nansum to ignore nan if exist
            grad[dim, i,:] = velocity_center

    # find the optimal location
    min_idx = np.zeros((3,3))
    for dim in range(3):
        for i in range(3):
            min_idx[dim, i] = np.argmin(grad[dim,:,i])

    optimal_shift = (np.mean(min_idx, axis=1) + 0.5).astype(int) # bigger than 0.5 take it, smaller than 0.5 igmaore
    
    # shift the rigidbody to neck 
    neckposition =  center_marker
    for i in range(3):
        offset = np.zeros(3)
        offset[dim] = 1
        offset_basis = rigid_body_basis @ offset
        temp = offset_basis * shift_range[optimal_shift[i]]
        neckposition = neckposition + temp
    
    print('neck position relative to the center marker is x : %.3f cm, y: %.3f cm, z: %.3f cm'%(shift_range[optimal_shift[0]]*100,shift_range[optimal_shift[1]]*100,shift_range[optimal_shift[2]]*100) )
    return neckposition    


def remove_outlier(marker_speed,speed_range = 0.1):
    '''
    find the correct range of idnex based on the speed
    speed_range: remove the period when the speed is bigger than the speed range(m/sec)
    '''
    speed_idx = np.unique(np.argwhere((marker_speed>speed_range) | (marker_speed< - speed_range) )[:,0]) # in shape n
    right_speed = np.delete(marker_speed, speed_idx, axis=0)
    return right_speed
        
    