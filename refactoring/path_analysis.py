import numpy as np

import base

##TODOs: implement ver.1 straightness function, bootstraping


def straightness_moment_time(trial_trajectory, before_time=3):
    def _straight_line(start, end, length):

        _x = np.linspace(start[0], end[0], length)
        _y = np.linspace(start[1], end[1], length)

        return np.vstack([_x, _y]).T

    def _straight_length(start, end):

        return np.sqrt(np.sum((start - end)**2))

    def travel_distance(traj):
        return np.cumsum(np.sqrt(np.sum((traj[1:] - traj[:-1])**2,
                                        axis=1)))[-1]

    time = trial_trajectory[:, 0]
    time_after = np.cumsum(np.flip(time[1:] - time[:-1]))
    if time_after[-1] > before_time:
        end_ind = np.where(time_after >= before_time)[0][0]
        trial_trajectory = np.flip(trial_trajectory[:, 1:3, ])[end_ind:]

    straight_length = _straight_length(trial_trajectory[0],
                                       trial_trajectory[-1])
    trajectory_displacement = travel_distance(trial_trajectory)
    straightness = straight_length / trajectory_displacement

    return straightness, np.array([trial_trajectory[0],
                                   trial_trajectory[-1]]), trial_trajectory


def straightness_over_time(trial_trajectory, before_time=2):

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

    time = trial_trajectory[:, 0]
    time_after = np.cumsum(np.flip(time[1:] - time[:-1]))
    if time_after[-1] > before_time:
        end_ind = np.where(time_after >= before_time)[0][0]
        trial_trajectory = np.flip(trial_trajectory[:, 1:3, ])[end_ind:]
    for i in range(len(trial_trajectory) - 1):
        straight_length = _straight_length(trial_trajectory[i],
                                           trial_trajectory[-1])
        trajectory_displacement = travel_distance(trial_trajectory[i:])

        straightness.append(straight_length / trajectory_displacement)

    return straightness, (trial_trajectory[0],
                          trial_trajectory[-1]), trial_trajectory
