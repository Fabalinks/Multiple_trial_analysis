import numpy as np
from utils.fix_jumps_utils import * 
from utils.transform_utils import *
import scipy.interpolate as interpolate
import pandas as pd
from utils.find_neck_utils import finding_neck

class SessionData(object):
    '''
    organize the data from raw file
    '''
    def __init__(self, file_path, stl_markers, sample_rate = 120):
        self.raw_data = pd.read_csv(file_path, skiprows=[0],index_col=None, header= [1,2,3,4])
        self.sample_num = len(self.raw_data)
        self.stl_markers = stl_markers/1000 # make it in meter unit
        self.sample_rate = sample_rate # frame frequency 120hz
        

    def init(self):
        '''
        generate basic feature of the session
        '''
        self.unit = 'meter' # unit used 
        self.markers = generate_marker_loc(range(0, self.sample_num), self.raw_data) # coords of four markers in time serires, np.array in shape [self.sample_num, 4, 3], in meter unit
        self.raw_quaternion = self.raw_data.iloc[:,34:38].values # numpy.array in [self.sample_num, 4], [qx, qy, qz, qw], contain the flip error
        
        # calculate the rotation angle of mouse, in shape [self.sample_num, 3], the angle is [roll, yaw, pitch]
        self.raw_angles = quaternion_to_euler(self.raw_quaternion[:,0], self.raw_quaternion[:,1],self.raw_quaternion[:,2],self.raw_quaternion[:,3]).T 
        
        self.origin, self.axis, self.mouse_coord = self.build_self_coordinate() # coordinates of markers when taking center middle marker as origin
        self.nan_idx = np.argwhere(np.isnan(self.markers[:,0,0])).reshape(-1) # nan idx along time series
        self.non_nan_idx = np.argwhere(~np.isnan(self.markers[:,0,0])).reshape(-1) # not nan idx along time series
        
        ## generate quality of marker
        self.marker_quality = np.zeros((self.sample_num,4))
        for i,idx  in enumerate([45,49,53,57]):
            self.marker_quality[:,i] = self.raw_data.iloc[:,idx]
        
    def build_self_coordinate(self):
        origin = self.stl_markers[3,:] # the center middle marker
        axis = np.eye(3) # the basis axis
        mouse_coord = determine_self_coor(origin, axis, self.stl_markers) # coordinates of markers when taking center middle marker as origin
        return origin, axis, mouse_coord
    
    
    def fix_flip(self, differential_thresh = 0.02,discontinu_tolerance = 10,basic_neighbor = 5, distance_neigobor = 20, inferred_range= 100, which_method = 'transition'):
        '''
        Fix the flipping of the marker, return the correted quaternion
        '''
        clean_markers = self.markers[self.non_nan_idx]
        jump_slice = detect_jumps(clean_markers,delta_threshold=differential_thresh, single_threshold=0.005)        
        smooth_loc,transition_direction = detect_transition(clean_markers,diff_thresh=differential_thresh,discontinu_tolerance = discontinu_tolerance, neighbor=basic_neighbor)
        
        # # get the fixed version of markers (corresponded to markers)
        if which_method == 'trainsition':
            fix_markers = fix_jumps_with_transition_state(clean_markers, jump_slice, smooth_loc= smooth_loc, transition_direction = transition_direction, neighbor=2)    
        elif which_method == 'distance':
            fix_markers= detect_jumps_with_distance(clean_markers, jump_slice, self.marker_quality[self.non_nan_idx], smooth_loc, neighbor= distance_neigobor,inferred_range=inferred_range)
        else:
            # sec_jump_slice = detect_jumps(fix_markers,delta_threshold=differential_thresh, single_threshold=0.005)          
            fix_markers = fix_jumps(clean_markers, jump_slice, neighbor= basic_neighbor)

        self.fix_markers = np.zeros_like(self.markers)
        self.fix_markers[self.nan_idx,:,:] = np.tile(np.nan,[len(self.nan_idx),4,3])
        self.fix_markers[self.non_nan_idx,:,:] = fix_markers
        

        self.flip_index = find_flip_index(self.fix_markers, self.markers)

        
        
        self.fix_quaternion = calcualte_angles_after_flip_correction(self.mouse_coord, self.fix_markers, self.markers, self.raw_quaternion)
        
        self.fix_angles = quaternion_to_euler(self.fix_quaternion[:,0], self.fix_quaternion[:,1], self.fix_quaternion[:,2], self.fix_quaternion[:,3]).T # in shape [self.sample_num, 3]
        return jump_slice

    
    def recheck_filp(self, differential_thresh = 0.02, neigobor = 3 ):
        
        self.recheck_markers = np.copy(self.fix_markers)
        
        abnormal_loc = np.argwhere((self.fix_angles[:,-1] >100)| (self.fix_angles[:,-1] <-100)).reshape(-1)
        # abnormal_loc_raw = np.argwhere((self.raw_angles[:,-1] >100)| (self.raw_angles[:,-1] <-100)).reshape(-1)
        wrong_flip_loc = np.intersect1d(abnormal_loc, self.flip_index)
        # real_wrong_loc = np.setdiff1d(wrong_flip_loc, abnormal_loc_raw)
        
        self.recheck_markers[wrong_flip_loc,0,:] = self.fix_markers[wrong_flip_loc,2,:]
        self.recheck_markers[wrong_flip_loc,2,:] = self.fix_markers[wrong_flip_loc,0,:] # re-reflip 
        
        self.re_flip_index = find_flip_index(self.recheck_markers, self.markers)
        
            
        self.re_fix_quaternion = calcualte_angles_after_flip_correction(self.mouse_coord, self.recheck_markers, self.markers, self.raw_quaternion)
        
        self.re_fix_angles = quaternion_to_euler(self.re_fix_quaternion[:,0], self.re_fix_quaternion[:,1], self.re_fix_quaternion[:,2], self.re_fix_quaternion[:,3]).T # in shape [self.sample_num, 3]
        return
    
    def interpolate_missing_data(self, neighbors = 2, gap_thresh = 5):
        '''
        Fill the missing data in small gap, leave the larger gap untouched
        
        gap_thresh: if the gap length bigger than gap_thresh, ignore it
        neighbors: data length used to interpolated 
        '''
        _, nan_slice = continuous_detection(self.nan_idx,discontinu_tolerance=1,total_tolerance= np.exp(10), seperate=True)
        interpolated_markers = np.copy(self.recheck_markers)
        
        for this_slice in tqdm(nan_slice): 
            this_len = len(this_slice)
            if this_len<gap_thresh: # only fill the nan for shorter gap 
                left_close_non_nan = np.argsort(np.abs(self.non_nan_idx - this_slice[0]))[0:neighbors]
                this_x = np.sort(self.non_nan_idx[left_close_non_nan])
                # this_x = np.sort(this_session.non_nan_idx[left_close_non_nan])
                this_y = self.recheck_markers[this_x]

                new_x = this_slice
                f = interpolate.BarycentricInterpolator(this_x, this_y, axis=0)
                interpolated_markers[this_slice] = f(new_x)

        self.interpolated_markers = interpolated_markers
        
        self.interpolated_quaternion = calcualte_angles_after_flip_correction(self.mouse_coord, self.interpolated_markers, self.markers, self.raw_quaternion,inclue_nan= True)
        
        self.interpolated_angles = quaternion_to_euler(self.interpolated_quaternion[:,0], self.interpolated_quaternion[:,1], self.interpolated_quaternion[:,2], self.interpolated_quaternion[:,3]).T # in shape [self.sample_num, 3]
        
        return
    
    def interpolate_missing_data(self, neighbors = 2, gap_thresh = 5):
        '''
        Fill the missing data in small gap, leave the larger gap untouched
        
        gap_thresh: if the gap length bigger than gap_thresh, ignore it
        neighbors: data length used to interpolated 
        '''
        _, nan_slice = continuous_detection(self.nan_idx,discontinu_tolerance=1,total_tolerance= np.exp(10), seperate=True)
        interpolated_markers = np.copy(self.recheck_markers)
        
        for this_slice in tqdm(nan_slice): 
            this_len = len(this_slice)
            if this_len<gap_thresh: # only fill the nan for shorter gap 
                left_close_non_nan = np.argsort(np.abs(self.non_nan_idx - this_slice[0]))[0:neighbors]
                this_x = np.sort(self.non_nan_idx[left_close_non_nan])
                # this_x = np.sort(this_session.non_nan_idx[left_close_non_nan])
                this_y = self.recheck_markers[this_x]

                new_x = this_slice
                f = interpolate.BarycentricInterpolator(this_x, this_y, axis=0)
                interpolated_markers[this_slice] = f(new_x)

        self.interpolated_markers = interpolated_markers
        
        self.interpolated_quaternion = calcualte_angles_after_flip_correction(self.mouse_coord, self.interpolated_markers, self.markers, self.raw_quaternion,inclue_nan= True)
        
        self.interpolated_angles = quaternion_to_euler(self.interpolated_quaternion[:,0], self.interpolated_quaternion[:,1], self.interpolated_quaternion[:,2], self.interpolated_quaternion[:,3]).T # in shape [self.sample_num, 3]
        
        return
    
    def generate_neckposition(self):
        '''generate the neck position with Justin's methods. Seems not fully work.
        '''
        rigid_body_basis = quaternion_to_rotation_matrix(self.re_fix_quaternion)
        use_index = np.argwhere((self.speed_3d>0.05) & (self.speed_3d<1)).reshape(-1) # in meter/sec, select appropriate range of data 
        
        center_marker = self.recheck_markers[:,-1,:] # marker 4
        neckposition = finding_neck(rigid_body_basis, center_marker, use_index)
        
        self.neckpostion = neckposition
        
        self.height = neckposition[:,1]
        self.speed_1d, self.speed_2d, self.speed_3d = calculate_speed(neckposition,sample_rate= self.sample_rate) # use the speed of the neckposition to replace the speed from the center marker
        return
        
    def generate_more_features(self,feature = 'recheck_markers'):
        '''
        Generate roll, yaw, pitch, speed with the fixed-flip marker
        '''
        
        if feature== 'fix_markers':
            if hasattr(self, 'fix_markers'): # only do it after fix the marker flip
                markers = self.fix_markers
                angles = self.fix_angles
            else:
                raise AttributeError('No fix_angle attribute, please run fix_flip function')
            
        elif feature=='recheck_markers':
            if hasattr(self, 'recheck_markers'): # only do it after fix the marker flip
                markers = self.recheck_markers
                angles = self.re_fix_angles
            else:
                raise AttributeError('No recheck fix_angle attribute, please run recheck_flip function')
        elif feature == 'interpolated_markers':
            if hasattr(self, 'interpolated_markers'):
                markers = self.interpolated_markers
                angles = self.interpolated_angles
            else:
                raise AttributeError('No interpolated_markers attribute, please run interpolate_missing_data')
        else:
            raise NameError('incorrect marker name, it should be interpolated_markers/fix_markers/recheck_markers')
        
        self.pitch = angles[:,-1]
        self.roll = angles[:,0]
        self.yaw = angles[:,1]
        
        # height 
        self.height = markers[:,-1, 1]
        # speed 
        self.speed_3d = np.sqrt(np.sum(np.square(np.roll(markers[:,-1,:], -1, axis = 0) - markers[:,-1,:],), axis = 1))*self.sample_rate # m/sec take the center-middle marker 
        self.speed_2d = np.sqrt(np.sum(np.square(np.roll(markers[:,-1,[0,2]], -1, axis = 0) -markers[:,-1,[0,2]],), axis = 1))*self.sample_rate # only consider the speed on the horizontal planar
        self.speed_1d = (np.roll(markers[:,-1,1], -1, axis = 0) -markers[:,-1,1]) * self.sample_rate
        return

def calculate_speed(neckposition,sample_rate = 120):
    '''
    calculate the speed given a marker position in time series
    return
    
    speed_1d: 
    '''
    speed_3d = np.sqrt(np.sum(np.square(np.roll(neckposition[:,:], -1, axis = 0) - neckposition[:,:],), axis = 1))*sample_rate # m/sec take the center-middle marker 
    speed_2d = np.sqrt(np.sum(np.square(np.roll(neckposition[:,[0,2]], -1, axis = 0) -neckposition[:,[0,2]],), axis = 1))*sample_rate # only consider the speed on the horizontal planar
    speed_1d = (np.roll(neckposition[:,1], -1, axis = 0) -neckposition[:,1]) * sample_rate
    
    return speed_1d, speed_2d, speed_3d