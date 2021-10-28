import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

## Crawling all data saved in Data folder


def read_data(root_path, tag):
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
    if beacon_data is None or position_data is None:
        ValueError(f'One of beacon or position data is missing in {tag}')
    return beacon_data, position_data


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


class BeaconPosition():
    def __init__(self, root_path, tag):
        self.beacon_data, self.posisiton_data = read_data(root_path, tag)
        self.beacon_data = beacon_data
        self.position_data = position_data
        self.trials_data = make_trials(beacon_data, position_data)

    def get_statistic(self):
        """
        Implement method to get a statistic for one day data
        
        """
        pass


class MultiDaysBeaconPosition():
    def __init__(self, root_path, tags):
        self.beacon_list, self.position_list = multiple_days_data(
            root_path, tags)
