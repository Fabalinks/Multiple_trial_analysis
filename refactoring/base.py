import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

## Crawling all data saved in Data folder


def read_data(root_path, tag, has_beacon=True):
    """
    root_path: root directory where data is ex) ../Data/Raw/
    tag: date_time_tag to search for beacon/position data ex) '20211028-0030984'
    
    Returns beacon, position pair data with the given tag in DataFrame 
    """

    beacon_paths = glob.glob(os.path.join(root_path, 'beacon*'))
    position_paths = glob.glob(os.path.join(root_path, 'position*'))
    beacon_data = None
    position_data = None
    for p in beacon_paths:
        if tag in p:
            beacon_data = pd.read_csv(p, sep=" ", header=None)
    for p in position_paths:
        if tag in p:
            position_data = pd.read_csv(p, sep=" ", header=None)
    if has_beacon and beacon_data is None or position_data is None:
        raise ValueError(f'One of beacon or position data is missing in {tag}')
    return position_data, beacon_data


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


def make_trials(beacon_data, position_data):
    """
    Implement a func. to partition full data into trials based on beacon trigger?
    """
    return None


def rotation_correction(position_data):

    alpha = (5) * np.pi / 180

    rot_position_data = position_data

    rot_position_data[:, 1] = position_data[:, 1] * np.cos(
        alpha) - position_data[:, 3] * np.sin(alpha)

    rot_position_data[:, 3] = position_data[:, 1] * np.sin(
        alpha) + position_data[:, 3] * np.cos(alpha)

    return rot_position_data


class BeaconPosition():
    def __init__(self, root_path, tag, has_beacon=True):
        self.position_data, self.beacon_data = read_data(
            root_path, tag, has_beacon)
        self.position_data = rotation_correction(self.position_data.to_numpy())
        if has_beacon:
            self.beacon_data = self.beacon_data.to_numpy()
            self.trials_data = make_trials(self.beacon_data,
                                           self.position_data)

        self.get_distance_speed()
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
        self.speed = (self.displacement / self.time_bin) * 100  #Check unit


class MultiDaysBeaconPosition():
    def __init__(self, root_path, tags, has_beacon):
        #self.beacon_list, self.position_list = multiple_days_data(
        #    root_path, tags)
        self.dataset_list = []
        for tag in tags:
            self.dataset_list.append(BeaconPosition(root_path, tag,
                                                    has_beacon))
        self.multisession_statistics, self.individual_statistics = self.get_statistics(
        )

    @property
    def num_sessions(self):
        return len(self.dataset_list)

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
