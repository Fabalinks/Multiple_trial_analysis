import numpy as np

import base


def straightness_over_time(trial_trajectory, from_start=200):
    straightness = []

    def _straight_line(start, end, length):

        _x = np.linspace(start[0], end[0], length)
        _y = np.linspace(start[1], end[1], length)

        return np.vstack([_x, _y]).T

    def _straight_length(start, end):

        return np.sqrt(np.sum((start - end)**2))

    def travel_distance(traj):
        return np.cumsum(np.sqrt(np.sum((traj[1:] - traj[:-1])**2,
                                        axis=1)))[-1]

    trial_trajectory = trial_trajectory[-from_start:]

    for i in range(1, from_start - 1):
        straight_length = _straight_length(trial_trajectory[i],
                                           trial_trajectory[-1])
        trajectory_displacement = travel_distance(trial_trajectory[i:])

        straightness.append(straight_length / trajectory_displacement)

    return straightness
