import os
import sys
import glob
import argparse
from this import d
import numpy as np
import pandas as pd

## Filtering of speed, position, **sampling rate differs across trials!

##Global variable for adjusting offsets between beacon position and rat position for FS04

x_max, x_min = 0.2, -0.76
x_offset = x_max - (x_max - x_min)/2
y_max, y_min = 0.072,  -1.754
y_offset = y_max - (y_max - y_min)/2

##Global variable for getting relevant arena 

X_cut_min = -.59
Y_cut_max = 1.61
X_cut_max = .12
Y_cut_min = .00

def read_data(root_path, tag, has_beacon=True, has_metadata=True):
    """
    root_path: root directory where data is ex) ../Data/Raw/
    tag: date_time_tag to search for beacon/position data ex) '20211028-0030984'
    
    Returns beacon, position pair data with the given tag in DataFrame 
    """

    beacon_paths = glob.glob(os.path.join(root_path, 'beacons*'))
    position_paths = glob.glob(os.path.join(root_path, 'position*'))
    meta_paths = glob.glob(os.path.join(root_path, 'meta*'))
    beacon_data = None
    position_data = None
    metadata = None
    for p in beacon_paths:
        if tag in p:
            beacon_data = pd.read_csv(p, sep=" ", header=None)
    for p in position_paths:
        if tag in p:
            position_data = pd.read_csv(p, sep=" ", header=None)
    for p in meta_paths:
        if tag in p:
            metadata = pd.read_csv(p, sep=" : ", header=None, index_col=0).T
    if has_beacon and beacon_data is None or position_data is None:
        raise ValueError(f'One of beacon or position data is missing in {tag}')
    if has_metadata and metadata is None:
        raise ValueError(f'Metadata is missing in {tag}')
    return position_data, beacon_data, metadata


def multiple_days_data(root_path, tags):
    """
    Simple extension of read_data function.
    Make list of beacon/position data pair of the given tags
    """
    beacon_datalist = []
    position_datalist = []
    for tag in tags:
        beacon_data, position_data = read_data(root_path, tag)
        beacon_datalist.append(beacon_data)
        position_datalist.appned(position_data)
    return beacon_datalist, position_datalist

def cumulative_angle(data):
    """
    Change 'yaw' to cumulative angle changes.
    """
    deltas = []
    data = data+180
    for n, (before,after) in enumerate(zip(data[:-1], data[1:])):
        delta = after - before
        if delta < -180:
            deltas.append(360 + delta)
        elif delta > 180:
            deltas.append(-360 + delta)      
        else:
            deltas.append(delta)
            
    return np.insert(np.cumsum(deltas), 0,0)

def make_trials(position_data, beacon_data, metadata, frequency = 50):
    '''
    Return
    trial_list: list of numpy arrays position data [time, x, y, z, x-rot, y-rot, z-rot] 
    trial_beacon: list of beacon position
    trial_visible: list of booleans referring visibility of trial (0 - visible, 1 - successful invisible, 2 - failed invisible )
    '''
    trial_list = []
    trial_beacon = beacon_data[:, -2:]
    start = position_data[0,0]
    end = position_data[-1,0]
    position_data=position_data[[np.argmin(abs(position_data[:,0]-t)) for t in np.arange(start, end+1/frequency, 1/frequency)]]
    position_data[:,0] = np.arange(start, end+1/frequency, 1/frequency)
    
    
    beacon_shown_time_idx = [
        np.argmin(abs(position_data[:, 0] - i)) for i in beacon_data[:, 0]
    ]

    trial_list.append(position_data[:beacon_shown_time_idx[0]])
    for i in range(len(beacon_shown_time_idx)):
        if i != len(beacon_shown_time_idx) - 1:
            trial = position_data[
                beacon_shown_time_idx[i]:beacon_shown_time_idx[i + 1]]
        else:
            trial = position_data[beacon_shown_time_idx[i]:]
        trial_list.append(trial)
    before_beacon = position_data[:beacon_shown_time_idx[0]]
    visible = []
    ## visible 0, invisible successful 1, invisible fail 2
    if metadata is None:
        return trial_list, trial_beacon

    elif 'invisible_time' in metadata.keys():
        trial_visible = np.zeros(len(trial_list))
        invisible_time = eval(metadata['invisible_time'].item())
        invisible_index = eval(metadata['invisible_list'].item())
        invisible_frequency = eval(metadata['light_off'].item())
        trial_visible[np.arange(len(trial_list))%invisible_frequency == (invisible_frequency-1)] = 2
        trial_visible[invisible_index] = 1
        """
        for n, trial in enumerate(trial_list):
            trial_visible_ = np.ones_like(trial)
            if n in invisible_index:
                time_after = np.cumsum(trial[1:, 0] - trial[:-1, 0])
                if time_after[-1] > invisible_time:
                    invisible_end = np.where(time_after >= invisible_time)[0][0]
                    trial_visible_ = trial_visible_[:invisible_end]
                trial_visible[n] = False
            visible.append(trial_visible_)
        """
        return trial_list, trial_beacon, trial_visible
    else:
        return trial_list, trial_beacon

   

    


def rotation_correction(position_data):

    alpha = (2) * np.pi / 180

    rot_position_data = position_data

    rot_position_data[:, 1] = position_data[:, 1] * np.cos(
        alpha) - position_data[:, 2] * np.sin(alpha)

    rot_position_data[:, 2] = position_data[:, 1] * np.sin(
        alpha) + position_data[:, 2] * np.cos(alpha)

    return rot_position_data


class BeaconPosition():
    """
    Single session data class

    """
    def __init__(self, root_path, tag, has_beacon=True, has_metadata=True):
        position_data, self.beacon_data, self.metadata = read_data(
            root_path, tag, has_beacon, has_metadata)
        
        if position_data.shape[1] == 4:
            """
            Case for old dataset where angle is not recorded
            """
            self.position_data = rotation_correction(position_data.to_numpy()[:, [0,1,3,2]])
        else:
            self.position_data=rotation_correction(position_data.to_numpy()[:, [0,1,3,2,4,5,6]])
            self.position_data[:,5]=cumulative_angle(self.position_data[:, 5])
        self.position_data[:,1] = self.position_data[:,1] - x_offset
        self.position_data[:,2] = self.position_data[:,2] + y_offset
        

        if has_beacon:
            self.beacon_data = self.beacon_data.to_numpy()
            self.beacon_data = rotation_correction(self.beacon_data[:, [0,-2,-1]])
            if not has_metadata:
                self.trial_list, self.trial_beacon = make_trials(self.position_data, self.beacon_data, None)
            else:
                self.trial_list, self.trial_beacon, self.trial_visible = make_trials(
                    self.position_data, self.beacon_data, self.metadata)
                self.total_trial = len(self.trial_visible)
                self.num_invisible_trial = (self.trial_visible != 0).sum()
                self.num_invisible_successful_trial = (self.trial_visible == 1).sum()
                self.num_invisible_failed_trial = (self.trial_visible ==2).sum()
                self.num_visible_trial = (self.trial_visible == 0).sum()

        self.get_distance_speed()
        self.get_head_rotation()
        self.statistics = self.get_statistic()
            

    def get_statistic(self):

        self.avg_speed = np.mean(self.speed)
        self.median_speed = np.median(self.speed)
        self.total_distance = self.travel_distance[-1]

        return {
            'avg_speed': self.avg_speed,
            'median_speed': self.median_speed,
            'distance': self.total_distance
        }

    def get_distance_speed(self):

        self.displacement = np.sqrt(
            np.sum((self.position_data[1:, 1:3] -
                    self.position_data[:-1, 1:3])**2,
                   axis=1))
        self.travel_distance = np.cumsum(self.displacement, axis=0)
        self.time_bin = self.position_data[1:, 0] - self.position_data[:-1, 0]
        self.speed = (self.displacement / self.time_bin)  #Check unit

    def get_head_rotation(self):
        self.angular_displacement = self.position_data[1:, 4:] - self.position_data[:-1, 4:]
        #self.angular_displacement[:, 1] = cumulative_angle(self.angular_displacement[:, 1])
        self.time_bin = self.position_data[1:, 0] - self.position_data[:-1, 0]
        self.angular_velocity = self.angular_displacement/self.time_bin[:, None]
        



class MultiDaysBeaconPosition():
    def __init__(self, root_paths, tags, has_beacon, has_metadata):
        #self.beacon_list, self.position_list = multiple_days_data(
        #    root_path, tags)
        self.dataset_list = []
        for root_path, tag in zip(root_paths, tags):
            self.dataset_list.append(
                BeaconPosition(root_path, tag, has_beacon, has_metadata))
        self.multisession_statistics, self.individual_statistics = self.get_statistics(
        )
        if has_beacon:
            self.get_trials()

    @property
    def num_sessions(self):
        return len(self.dataset_list)

    def get_trials(self):
        self.trial_list = []
        self.beacon_list = []
        self.trial_visible = []
        #self.visible = []
        for session in self.dataset_list:
            self.trial_list.append(session.trial_list)
            #self.visible.append(session.visible)
            self.beacon_list.append(session.trial_beacon)
            if hasattr(session,'trial_visible'):
                self.trial_visible.append(session.trial_visible)

    def get_statistics(self):
        ## Across session statistics
        self.avg_speed = np.mean(
            [session.avg_speed for session in self.dataset_list])
        self.avg_distance = np.mean(
            [session.total_distance for session in self.dataset_list])
        self.median_speed = np.median(
            np.concatenate([session.speed for session in self.dataset_list]))

        multisession_statistics = {
            'avg_speed': self.avg_speed,
            'avg_distance': self.avg_distance,
            'median_speed': self.median_speed
        }

        ##Individual session statistics
        self.avg_speed_per_session = [
            session.avg_speed for session in self.dataset_list
        ]
        self.median_speed_per_session = [
            session.median_speed for session in self.dataset_list
        ]
        self.distance_per_session = [
            session.total_distance for session in self.dataset_list
        ]
        individual_statistics = {
            'avg_speeds': self.avg_speed_per_session,
            'distances': self.distance_per_session,
            'median_speeds': self.median_speed_per_session
        }

        return multisession_statistics, individual_statistics

    def get_rearings(self, threshold= 0.65):
        
        """
        Find rearings, which is defined as an event with z position above threshold in each trial
        Return: counts of rears in each trial, times of rears, x,y position of rears in each trial and session.

        """
        session_rearing = []
        session_rearing_counts =[]
        session_rearing_durations = []
        session_rearing_distance = []

        def rear_in_arena(data):
            return (data[:,1]> X_cut_min) & (data[:,1]< X_cut_max) & (data[:,2]>Y_cut_min) & (data[:,2]<Y_cut_max)

        for i, session in enumerate(self.trial_list):
            trial_rearings = []
            trial_rearing_counts = []
            trial_rearing_durations = []
            trial_distance_beacon =[]
            for j, trial in enumerate(session[1:]):
                rearings = (trial[:, -1] >= threshold) & (rear_in_arena(trial))
                events =np.argwhere(np.diff(rearings))
                events = events.reshape(len(events)).tolist()
                trial_rearings.append(trial[rearings])
                ## get counts of rearing events wihtin a trial 
                if len(events)!=0 and rearings[0]:
                    counts = int(len(events)//2) + 1
                else:
                    counts = int(len(events)//2) + len(events)%2
                trial_rearing_counts.append(counts)
                
                ## count time for each rear
                if len(rearings)>0:
                    if rearings[0] and len(events)%2 ==0:
                        starts = ([0] + events)[0::2]
                        ends = (events+[-1])[0::2]
                    elif rearings[0] and len(events)%2 ==1:
                        starts = ([0]+ events)[0::2]
                        ends = events[0::2]
                    elif (not rearings[0] and len(events)%2 ==0):
                        starts = events[0::2]
                        ends = events[1::2]
                    elif (not rearings[0] and len(events)%2==1):
                        starts = events[0::2]
                        ends = (events+ [-1])[1::2]
                    rearing_duration = [trial[e,0]-trial[s,0] for s, e in zip(starts, ends)]
                    beacon_pos = self.beacon_list[i][j]
                    distance_from_beacon = [np.mean(np.linalg.norm(trial[s:e, 1:3] - beacon_pos, axis = 1)) for s,e in zip(starts, ends)]
                    trial_rearing_durations.append(rearing_duration)
                    trial_distance_beacon.append(distance_from_beacon)
                ## measure a distance from beacon while rearing

            session_rearing.append(trial_rearings)
            session_rearing_counts.append(trial_rearing_counts)
            session_rearing_durations.append(trial_rearing_durations)
            session_rearing_distance.append(trial_distance_beacon)
        
        return session_rearing, session_rearing_counts, session_rearing_durations, session_rearing_distance
