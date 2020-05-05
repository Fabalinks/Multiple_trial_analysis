import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.tri as tri
import numpy as np
from scipy.spatial.transform import Rotation as R
from statistics import mean
import math
import time

root = 'C:/Users/Fabian/Desktop/Analysis/Multiple_trial_analysis/Data/Raw/'
figures = 'C:/Users/Fabian/Desktop/Analysis/Multiple_trial_analysis/Figures/'

#Data with beacon metadata
beacon = pd.read_csv(root+'beacons 20200128-151826.txt',sep=" ", header=None)
beacon2 = pd.read_csv(root+'beacons 20200128-160013.txt',sep=" ", header=None)

#Data with beacon metadata

beacon_Day86_fs2 = pd.read_csv(root+'beacons 20200128-151826.txt',sep=" ", header=None)
beacon_Day86_fs1 = pd.read_csv(root+'beacons 20200128-160013.txt',sep=" ", header=None)

beacon_Day87_fs2 = pd.read_csv(root+'beacons 20200129-153534.txt',sep=" ", header=None)
beacon_Day87_fs1 = pd.read_csv(root+'beacons 20200129-161806.txt',sep=" ", header=None)

beacon_Day88_fs2 = pd.read_csv(root+'beacons 20200130-102126.txt',sep=" ", header=None)
beacon_Day88_fs1 = pd.read_csv(root+'beacons 20200130-111741.txt',sep=" ", header=None)

beacon_Day89_fs2 = pd.read_csv(root+'beacons 20200130-161126.txt',sep=" ", header=None)
beacon_Day89_fs1 = pd.read_csv(root+'beacons 20200130-151829.txt',sep=" ", header=None)

beacon_Day90_fs2 = pd.read_csv(root+'beacons 20200203-154441.txt',sep=" ", header=None)
beacon_Day90_fs1 = pd.read_csv(root+'beacons 20200203-145842.txt',sep=" ", header=None)

beacon_Day91_fs2 = pd.read_csv(root+'beacons 20200204-125552.txt',sep=" ", header=None)
beacon_Day91_fs1 = pd.read_csv(root+'beacons 20200204-133905.txt',sep=" ", header=None)

beacon_Day92_fs2 = pd.read_csv(root+'beacons 20200205-143220.txt',sep=" ", header=None)
beacon_Day92_fs1 = pd.read_csv(root+'beacons 20200205-151052.txt',sep=" ", header=None)

beacon_Day93_fs2 = pd.read_csv(root+'beacons 20200206-133529.txt',sep=" ", header=None)
beacon_Day93_fs1 = pd.read_csv(root+'beacons 20200206-125706.txt',sep=" ", header=None)


Day46_fs1 = pd.read_csv(root+'position 20190923-174441.txt',sep=" ", header=None)
Day46_fs2 = pd.read_csv(root+'position 20190923-171112.txt',sep=" ", header=None)
Day47_fs1 = pd.read_csv(root+'position 20191001-112411.txt',sep=" ", header=None)
Day47_fs2 = pd.read_csv(root+'position 20191001-115127.txt',sep=" ", header=None)
Day48_fs1 = pd.read_csv(root+'position 20191002-115000.txt',sep=" ", header=None)
Day48_fs2 = pd.read_csv(root+'position 20191002-111038.txt',sep=" ", header=None)
Day51_fs1 = pd.read_csv(root+'position 20191106-170809.txt',sep=" ", header=None)
Day52_fs2 = pd.read_csv(root+'position 20191107-174215.txt',sep=" ", header=None)
Day52_fs1 = pd.read_csv(root+'position 20191107-183857.txt',sep=" ", header=None)
Day53_fs2 = pd.read_csv(root+'position 20191108-142321.txt',sep=" ", header=None)
Day53_fs1 = pd.read_csv(root+'position 20191108-145125.txt',sep=" ", header=None)
Day66_fs1 = pd.read_csv(root+'position 20191118-161325.txt',sep=" ", header=None)
Day66_fs2 = pd.read_csv(root+'position 20191118-171209.txt',sep=" ", header=None)
Day72_fs1 = pd.read_csv(root+'position 20191127-122008.txt',sep=" ", header=None)
Day72_fs2 = pd.read_csv(root+'position 20191127-132223.txt',sep=" ", header=None)


Day79_fs2 = pd.read_csv(root+'position 20200121-154004.txt',sep=" ", header=None)
Day79_fs1 = pd.read_csv(root+'position 20200121-161359.txt',sep=" ", header=None)

Day80_fs2 = pd.read_csv(root+'position 20200122-141738.txt',sep=" ", header=None)
Day80_fs1 = pd.read_csv(root+'position 20200122-133022.txt',sep=" ", header=None)

Day81_fs2 = pd.read_csv(root+'position 20200123-141930.txt',sep=" ", header=None)
Day81_fs1 = pd.read_csv(root+'position 20200123-150059.txt',sep=" ", header=None)

Day82_fs2 = pd.read_csv(root+'position 20200124-151642.txt',sep=" ", header=None)
Day82_fs1 = pd.read_csv(root+'position 20200124-160826.txt',sep=" ", header=None)

Day83_fs2 = pd.read_csv(root+'position 20200126-183810.txt',sep=" ", header=None)
Day83_fs1 = pd.read_csv(root+'position 20200126-180200.txt',sep=" ", header=None)

Day84_fs2 = pd.read_csv(root+'position 20200127-205615.txt',sep=" ", header=None)
Day84_fs1 = pd.read_csv(root+'position 20200127-155645.txt',sep=" ", header=None)

Day85_fs2 = pd.read_csv(root+'position 20200128-112255.txt',sep=" ", header=None)
Day85_fs1 = pd.read_csv(root+'position 20200128-104637.txt',sep=" ", header=None)

Day86_fs2 = pd.read_csv(root+'position 20200128-160013.txt',sep=" ", header=None)
Day86_fs1 = pd.read_csv(root+'position 20200128-151826.txt',sep=" ", header=None)

Day87_fs2 = pd.read_csv(root+'position 20200129-153534.txt',sep=" ", header=None)
Day87_fs1 = pd.read_csv(root+'position 20200129-161806.txt',sep=" ", header=None)

Day88_fs2 = pd.read_csv(root+'position 20200130-102126.txt',sep=" ", header=None)
Day88_fs1 = pd.read_csv(root+'position 20200130-111741.txt',sep=" ", header=None)

Day89_fs2 = pd.read_csv(root+'position 20200130-161126.txt',sep=" ", header=None)
Day89_fs1 = pd.read_csv(root+'position 20200130-151829.txt',sep=" ", header=None)

Day90_fs2 = pd.read_csv(root+'position 20200203-154441.txt',sep=" ", header=None)
Day90_fs1 = pd.read_csv(root+'position 20200203-145842.txt',sep=" ", header=None)

Day91_fs2 = pd.read_csv(root+'position 20200204-125552.txt',sep=" ", header=None)
Day91_fs1 = pd.read_csv(root+'position 20200204-133905.txt',sep=" ", header=None)

Day92_fs2 = pd.read_csv(root+'position 20200205-143220.txt',sep=" ", header=None)
Day92_fs1 = pd.read_csv(root+'position 20200205-151052.txt',sep=" ", header=None)

Day93_fs2 = pd.read_csv(root+'position 20200206-133529.txt',sep=" ", header=None)
Day93_fs1 = pd.read_csv(root+'position 20200206-125706.txt',sep=" ", header=None)




class Data_analysis:
    intiated = time.time()
    last_update = "3.5.2020"

    def calculate_Distance(x, y):
        """Calculates distance given position"""
        travel = 0
        for i in range(len(y) - 1):
            dist = math.sqrt((x[0 + i] - x[1 + i]) ** 2 + (y[0 + i] - y[1 + i]) ** 2)
            travel += dist

        return travel

    def calculate_median_Speed(x, y, time):
        """Calculates mean speed given position"""
        travel = 0
        speed = []
        for i in range(len(y) - 1):
            dist = math.sqrt((x[0 + i] - x[1 + i]) ** 2 + (y[0 + i] - y[1 + i]) ** 2) / time
            speed.append(dist)
        return (np.median(speed))

    def Distance_over_days(list_of_fs1_days, list_of_fs2_days, list_of_number_of_days):
        """this function takes lists of days for each animal and plots a distance covered over time. """

        LT_distance_fs1 = []
        for day in list_of_fs1_days:
            LT_distance_fs1.append(calculateDistance(list(day[1]), list(day[3])))

        LT_distance_fs2 = []
        for day in list_of_fs2_days:
            LT_distance_fs2.append(calculateDistance(list(day[1]), list(day[3])))

        x = np.arange(len(list_of_number_of_days))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(dpi=1000)
        FS1 = ax.bar(x - width / 2, LT_distance_fs1, width, label='FS1')
        FS2 = ax.bar(x + width / 2, LT_distance_fs2, width, label='FS2')

        ax.set_ylabel('meters')
        ax.set_xlabel('Day')
        ax.set_title('Total distance by animal')
        ax.set_xticks(x)
        ax.set_xticklabels(list_of_number_of_days)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(int(height)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(FS1)
        autolabel(FS2)

        fig.tight_layout()
        plt.savefig(figures + 'distance_over_days_' + Day_number_list[0] + '-' + Day_number_list[-1] + '.png', dpi=1000)
        plt.show()

    def Speed_over_days_mean(list_of_fs1_days, list_of_fs2_days, list_of_number_of_days):
        """this function takes lists of days for each animal and plots the averge speed for a day over given days. """

        def calculateSpeed(x, y, time):
            travel = 0
            speed = []
            for i in range(len(y) - 1):
                dist = math.sqrt((x[0 + i] - x[1 + i]) ** 2 + (y[0 + i] - y[1 + i]) ** 2) / time
                speed.append(dist)
            return (np.mean(speed))

        LT_distance_fs1 = []
        for day in list_of_fs1_days:
            LT_distance_fs1.append(100 * calculateSpeed(list(day[1]), list(day[3]), 0.01))  # *100 to get to cm

        LT_distance_fs2 = []
        for day in list_of_fs2_days:
            LT_distance_fs2.append(100 * calculateSpeed(list(day[1]), list(day[3]), 0.01))

        x = np.arange(len(list_of_number_of_days))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(dpi=250)
        FS1 = ax.bar(x - width / 2, LT_distance_fs1, width, label='FS1')
        FS2 = ax.bar(x + width / 2, LT_distance_fs2, width, label='FS2')

        ax.set_ylabel('centimeters/minute')
        ax.set_xlabel('Day')
        ax.set_title('Total average speed by animal')
        ax.set_xticks(x)
        ax.set_xticklabels(list_of_number_of_days)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(int(height)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(FS1)
        autolabel(FS2)

        fig.tight_layout()
        plt.savefig(figures + 'speed_over_days_mean_' + Day_number_list[0] + '-' + Day_number_list[-1] + '.png',
                    dpi=1000)
        plt.show()

    def Speed_over_days_median(list_of_fs1_days, list_of_fs2_days, list_of_number_of_days):
        """this function takes lists of days for each animal and plots the averge speed for a day over given days. """

        def calculate_median_Speed(x, y, time):
            """Calculates mean speed given position"""
            travel = 0
            speed = []
            for i in range(len(y) - 1):
                dist = math.sqrt((x[0 + i] - x[1 + i]) ** 2 + (y[0 + i] - y[1 + i]) ** 2) / time
                speed.append(dist)
            return (np.median(speed))

        LT_distance_fs1 = []
        for day in list_of_fs1_days:
            LT_distance_fs1.append(100 * calculate_median_Speed(list(day[1]), list(day[3]), 0.01))  # *100 to get to cm

        LT_distance_fs2 = []
        for day in list_of_fs2_days:
            LT_distance_fs2.append(100 * calculate_median_Speed(list(day[1]), list(day[3]), 0.01))

        x = np.arange(len(list_of_number_of_days))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(dpi=250)
        FS1 = ax.bar(x - width / 2, LT_distance_fs1, width, label='FS1')
        FS2 = ax.bar(x + width / 2, LT_distance_fs2, width, label='FS2')

        ax.set_ylabel('centimeters/minute')
        ax.set_xlabel('Day')
        ax.set_title('Total average speed by animal')
        ax.set_xticks(x)
        ax.set_xticklabels(list_of_number_of_days)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(int(height)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(FS1)
        autolabel(FS2)

        fig.tight_layout()
        plt.savefig(figures + 'speed_over_days_mean_' + Day_number_list[0] + '-' + Day_number_list[-1] + '.png',
                    dpi=1000)
        plt.show()

    def rotation_correction(Day_fs1, Day_fs2, day):
        """Plots occupancy in corrected x/y scales"""
        alpha = (5) * np.pi / 180
        Day_fs1x = Day_fs1[1] * np.cos(alpha) - Day_fs1[3] * np.sin(alpha)
        Day_fs1y = Day_fs1[1] * np.sin(alpha) + Day_fs1[3] * np.cos(alpha)
        Day_fs2x = Day_fs2[1] * np.cos(alpha) - Day_fs2[3] * np.sin(alpha)
        Day_fs2y = Day_fs2[1] * np.sin(alpha) + Day_fs2[3] * np.cos(alpha)

        fig, ax = plt.subplots(1, 2)
        # fig.set_size_inches( 7.2/2,16.2/2)
        plt.xticks([])

        ax[0].hist2d(Day_fs1x, Day_fs1y, bins=30, cmap='terrain', cmax=1000)
        ax[0].plot(Day_fs1x, Day_fs1y, color='olive', alpha=.6)

        ax[1].hist2d(Day_fs2x, Day_fs2y, bins=30, cmap='terrain', cmax=1000)
        ax[1].plot(Day_fs2x, Day_fs2y, color='olive', alpha=.6)

        # ax[0].add_patch(mpl.patches.Circle((-.3429, 0.2),.15,edgecolor='red', fill = False,linestyle='--',linewidth=2))
        # ax[1].add_patch(mpl.patches.Circle((-.3429, 0.2),.15,edgecolor='red', fill = False,linestyle='--',linewidth=2))
        ax[0].set_title('FS1', fontsize=10)
        ax[1].set_title('FS2', fontsize=10)
        ax[0].set_ylabel('Day %s ' % day)
        ax[0].set_xticks([])
        ax[1].set_xticks([])

        fig.dpi = 200
        plt.tight_layout()
        plt.show()

    def Total_beacons_over_sessions(list_of_beacon_days, list_of_number_of_days, animal_ID):
        """this function takes lists of days for each animal and plots a distance covered over time. """

        visible = []
        invisible = []

        for beacon in list_of_beacon_days:
            diff = np.diff(beacon[0])
            visible.append(sum(diff[1::2]))
            invisible.append(sum(diff[0::2]))

        x = np.arange(len(list_of_number_of_days))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(dpi=500)
        FS1 = ax.bar(x - width / 2, visible, width, label='visible ' + str(int(np.sum(visible) / 60)) + ' min')
        FS2 = ax.bar(x + width / 2, invisible, width, fill=False,
                     label='invisible ' + str(int(np.sum(invisible) / 60)) + ' min')

        ax.set_ylabel('seconds looking for beacon')
        ax.set_xlabel('session#')
        ax.set_title('Time to get beacon over session ' + animal_ID)
        ax.set_xticks(x)
        ax.set_xticklabels(list_of_number_of_days)
        ax.legend(loc='upper left', prop={'size': 5})

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(int(height)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 0),  # 3 points vertical offset - set to 0
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(FS1)
        autolabel(FS2)

        fig.tight_layout()
        plt.savefig(
            figures + 'Total_beacons_over_session_' + animal_ID + '_' + Day_number_list[0] + '-' + Day_number_list[
                -1] + '.png', dpi=1000)
        plt.show()
