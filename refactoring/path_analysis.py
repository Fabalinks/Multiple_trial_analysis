import numpy as np

import base


def straightness_moment_time(trial_trajectory, time_window=3):
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
    trajectory_ = np.flip(trial_trajectory[:, 1:3, ],
                            axis=0)

    if time_window is None:
        end_ind = -1
        trial_trajectory = trajectory_[:end_ind]
    elif time_after[-1] > time_window:
        end_ind = np.where(time_after >= time_window)[0][0]
        trial_trajectory = trajectory_[:end_ind]
    else:
        print('Time window exceeds the length of trajectory for this trial')
        return None

    straight_length = _straight_length(trial_trajectory[0],
                                       trial_trajectory[-1])
    trajectory_displacement = travel_distance(trial_trajectory)
    straightness = straight_length / trajectory_displacement

    return np.flip(straightness), np.array([trial_trajectory[0],
                                   trial_trajectory[-1]]), trial_trajectory


def straightness_over_time(trial_trajectory, time_window=2):

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
    trajectory_ = np.flip(trial_trajectory[:, 1:3, ],
                            axis=0)

    if time_window is None:
        end_ind = -1
        trial_trajectory = trajectory_[:end_ind]
    elif time_after[-1] > time_window:
        end_ind = np.where(time_after > time_window)[0][0]
        trial_trajectory = trajectory_[:end_ind]
    else:
        print('Time window exceeds the length of trajectory for this trial')
        # return None


    for i in range(len(trial_trajectory) - 2):
        straight_length = _straight_length(trial_trajectory[i + 1],
                                           trial_trajectory[0])
        trajectory_displacement = travel_distance(trial_trajectory[:i + 2])

        straightness.append(straight_length / trajectory_displacement)

    return np.flip(straightness), (trial_trajectory[0],
                          trial_trajectory[-1]), trial_trajectory


def bootstrap(trajectory,
              time_window=2,
              num_sampling=10,
              straightness_type='sliding'):
    ##trajectory = [time,x,y,z]
    ## sample sham trajectory from entire session randomly
    sham_straightness_list = []
    shuffle_count = 1
    while shuffle_count < num_sampling:
        random_index = np.random.randint(0, len(trajectory))
        time = trajectory[random_index:, 0]
        time_cum = np.cumsum(time[1:] - time[:-1])
        if time_cum[-1] > time_window:
            end_index = np.where(time_cum > time_window)[0][0]
            shuffle_count += 1

        else:
            continue
        sampled_trajectory = trajectory[random_index:random_index + end_index +
                                        100]
        if straightness_type == 'sliding':
            sham_straightness = straightness_over_time(
                sampled_trajectory, time_window)[0]

        elif straightness_type == 'fixed':
            sham_straightness = straightness_moment_time(
                sampled_trajectory, time_window)[0]

        sham_straightness_list.append(sham_straightness)
    return sham_straightness_list
