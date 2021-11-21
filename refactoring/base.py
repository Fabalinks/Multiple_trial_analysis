import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

## Filtering of speed, position, **sampling rate differs across trials!


def read_data(root_path, tag, has_beacon=True, has_metadata=True):
    """
    root_path: root directory where data is ex) ../Data/Raw/
    tag: date_time_tag to search for beacon/position data ex) '20211028-0030984'
    
    Returns beacon, position pair data with the given tag in DataFrame 
    """

    beacon_paths = glob.glob(os.path.join(root_path, 'beacon*'))
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
        position_datallist.appned(position_data)
    return beacon_datalist, position_datalist


def make_trials(position_data, beacon_data, metadata):
    '''
    Return
    trial_list: list of numpy arrays position data [time, x, y, z] 
    trial_beacon: list of beacon position
    trial_visible: list of booleans referring visibility of trial (True - visible, False- invisible
    before_beacon: list of positions before the first beacon appears
    '''

    trial_list = []
    trial_beacon = beacon_data[:, -2:]
    sampling_rate = position_data[1, 0] - position_data[0, 0]
    beacon_shown_time_idx = [
        np.argmin(abs(position_data[:, 0] - i)) for i in beacon_data[:, 0]
    ]
    for i in range(len(beacon_shown_time_idx)):
        if i != len(beacon_shown_time_idx) - 1:
            trial = position_data[
                beacon_shown_time_idx[i]:beacon_shown_time_idx[i + 1]]
        else:
            trial = position_data[beacon_shown_time_idx[i]:]
        trial_list.append(trial)
    before_beacon = position_data[:beacon_shown_time_idx[0]]
    visible = []
    trial_visible = [True] * len(trial_list)
    invisible_time = eval(metadata['invisible_time'].item())
    invisible_index = eval(metadata['invisible_list'].item())

    for n, trial in enumerate(trial_list):
        trial_visible_ = np.ones_like(trial)
        if n in invisible_index:
            time_after = np.cumsum(trial[1:, 0] - trial[:-1, 0])
            if time_after[-1] > invisible_time:
                invisible_end = np.where(time_after >= invisible_time)[0][0]
                trial_visible_ = trial_visible_[:invisible_end]
            trial_visible[n] = False
        visible.append(trial_visible_)

    return trial_list, trial_beacon, visible, trial_visible


def rotation_correction(position_data):

    alpha = (5) * np.pi / 180

    rot_position_data = position_data

    rot_position_data[:, 1] = position_data[:, 1] * np.cos(
        alpha) - position_data[:, 3] * np.sin(alpha)

    rot_position_data[:, 3] = position_data[:, 1] * np.sin(
        alpha) + position_data[:, 3] * np.cos(alpha)

    return rot_position_data


class BeaconPosition():
    """
    Single session data class
    ##TODO: extend trial_visible into time base, not trial base
    """
    def __init__(self, root_path, tag, has_beacon=True, has_metadata=True):
        self.position_data, self.beacon_data, self.metadata = read_data(
            root_path, tag, has_beacon, has_metadata)
        self.position_data = rotation_correction(
            self.position_data.to_numpy()[:, :4])
        if has_beacon:
            self.beacon_data = self.beacon_data.to_numpy()

        self.get_distance_speed()
        self.statistics = self.get_statistic()
        if has_beacon:
            self.trial_list, self.trial_beacon, self.visible, self.trial_visible = make_trials(
                self.position_data, self.beacon_data, self.metadata)

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
        self.speed = (self.displacement / self.time_bin) * 100  #Check unit


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
        self.visible = []
        for session in self.dataset_list:
            self.trial_list.append(session.trial_list)
            self.trial_visible.append(session.trial_visible)
            self.visible.append(session.visible)
            self.beacon_list.append(session.trial_beacon)

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
