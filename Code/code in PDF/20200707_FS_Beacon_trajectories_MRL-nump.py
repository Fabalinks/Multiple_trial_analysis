#!/usr/bin/env python
# coding: utf-8

# # Notebook for calculating the resultant lenght or MRL of beacon 

# 1. import functions
# 2. Create combined 3D vector (XYZ, #beacons, time-secodns before beacon) 
# 3. Create and plot resultant lenght 
# 4. Combine resultant lenghts into one array for division (short,long)
# 5. Plot ratios as a histogram 
# 6. Combine into a function 
# 7. Calculate histogram over sessions
# 8. Make into function 
# 9. Compute Sham ( bootstrap like) comparison 
# 10. Plot histograms with sham medians on top of original data 
# 11. Compute sliding median and mean window - over time - plot differences with sham 
# 12. Bar plot 

# In[3]:


import math
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from numpy import median
from scipy.stats import ranksums

root = 'C:/Users/Fabian/Desktop/Analysis/Multiple_trial_analysis/Data/Raw/'
figures = 'C:/Users/Fabian/Desktop/Analysis/Multiple_trial_analysis/Figures/'


# ## 1. Import functions from previous notebooks - 
# Giving trajectories before beacon can be improved to have the whole trajectory before beacon to the time when another beacon is reached - which will result in uneven lenghts and can make that as list of arrays, but numpy is not made for that. 

# In[4]:


from Data_analysis import *


# In[5]:


Day86_fs2 = pd.read_csv(root+'position 20200128-160013.txt',sep=" ", header=None)
Day86_fs1 = pd.read_csv(root+'position 20200128-151826.txt',sep=" ", header=None)

beacon_Day86_fs2 = pd.read_csv(root+'beacons 20200128-160013.txt',sep=" ", header=None)
beacon_Day86_fs1 = pd.read_csv(root+'beacons 20200128-151826.txt',sep=" ", header=None)

beacon_data = beacon_Day86_fs1
position_data = Day86_fs1

dist=[]
norm_x,norm_y,norm_time = position_before_beacon_trigger_beacon(5,beacon_data,position_data)
for i in np.arange(len(norm_x)):
    plt.plot(norm_x[i],norm_y[i])
    #dist.append(calculate_Distance(norm_x[i],norm_y[i])) - cannot be indexed into ... 
    


# ## 2. create combined vector lenght from trajectory,  can make it normalized if needed. 
# 

# #### A. Convert everything to Numpy - previously as a padnas dataframe - no need untill dedling with non-numbers 

# In[6]:


beacon_d = beacon_data.to_numpy()
pos_data = position_data.to_numpy()
beacon_d[:, 0] -= pos_data[0][0]
pos_data[:, 0] -= pos_data[0][0]


# #### B. Get index where beacon was reached. 

# In[7]:


def get_index_at_pos(beacon_data, position_data):
    indexes = []
    for beacon_t in beacon_data[:, 0]:
        indexes.append( np.abs(beacon_t - position_data[:, 0]).argmin() )
        
    return indexes


# #### C. Seperate function to create arrays of positions before beacon reached. - this gives different size of position data hence it cannot be an array, but a lits of arrays. 

# In[8]:


def get_positions_before(seconds_back, idxs, position_data):
    """create arrays of positions before beacon reached"""
    beacon_periods = []
    for beacon_idx in idxs:
        beacon_t = position_data[beacon_idx][0]
        beacon_t_before = beacon_t - seconds_back
        before_idx = np.abs(beacon_t_before - position_data[:, 0]).argmin()
        beacon_periods.append(position_data[before_idx:beacon_idx])
        
    return beacon_periods


# In[9]:


idxs = get_index_at_pos(beacon_d, pos_data)
periods = get_positions_before(3.5, idxs, pos_data)
print(np.array(periods).shape)
print(periods[1].shape)
print(periods[10].shape)
beacon_travel=periods


# ### Looking at time if it is the correct array... now corrected

# In[10]:


print(beacon_travel[0][209][0])
diff=[]
for i in range (209):
    diff.append(beacon_travel[1][i][0]-beacon_travel[1][i+1][0])
    
(sum(diff)) # so it is giving 5 seconds worth of trajectory? 
beacon_travel[0][:,1]


# ### Sanity check - checkt trajectories and how they look in real - if using indexes it is a bit longer... 

# In[11]:


seconds_back = 3
index, enum  = get_index(beacon_data, position_data)
plt.plot((position_data[1][index[1]-(seconds_back*100):index[1]]),(position_data[3][index[1]-(seconds_back*100):index[1]]))
plt.plot(beacon_travel[1][:,1],beacon_travel[1][:,3]) # plotting trajectory of the second beacon...


# In[12]:


beacon_travel[1][:,1]


# ## 3. create lenght of trajectory form start to end - i.e straight line 

# In[13]:


straights=[]
longs=[]
for beacon in range(len(beacon_travel)):
    longs.append(calculate_Distance(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3]))
    straights.append(math.sqrt((beacon_travel[beacon][0,1] - beacon_travel[beacon][-1,1]) ** 2 + (beacon_travel[beacon][0,3] - beacon_travel[beacon][-1,3]) ** 2))
print(longs)
print(straights)


# In[14]:


beacon = 4
plt.plot(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3])
plt.plot((beacon_travel[beacon][0,1] ,beacon_travel[beacon][-1,1]) ,(beacon_travel[beacon][0,3] , beacon_travel[beacon][-1,3]))
    


# ### Sanity check: plotting the resultant lenght! - only first 16 in this case

# In[15]:


fig,ax = plt.subplots(4,4,figsize=(20,20),dpi=200)
num=0
h=0
for beacon in range(16): 
    s=(math.sqrt((beacon_travel[beacon][0,1] - beacon_travel[beacon][-1,1]) ** 2 + (beacon_travel[beacon][0,3] - beacon_travel[beacon][-1,3]) ** 2))
    
    l=calculate_Distance(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3])
    
    ax[h][num].plot(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3])
    ax[h][num].plot((beacon_travel[beacon][0,1] ,beacon_travel[beacon][-1,1]) ,(beacon_travel[beacon][0,3] , beacon_travel[beacon][-1,3]))
    
    
    distl = mlines.Line2D([], [], marker='_',markersize=5,markerfacecolor="blue",
                            markeredgecolor="blue",linewidth = 0, label='Distance long %.2f m' %l)
    dists = mlines.Line2D([], [], marker='_',markersize=5,markerfacecolor="orange",
                            markeredgecolor="orange",linewidth = 0, label='Distance short %.2f m' %s)
    diff = mlines.Line2D([], [], marker=" ",linewidth = 0, label='Ratio = %.2f ' %(s/l))
    
        
    ax[h][num].legend(handles=[distl,dists,diff],loc='best',prop={'size': 6})
    
    l=0
    s=0
    h+=1
    if h % 4==0:
        num += 1
        h=0
plt.savefig('%s16_trajectories_2_sec._before_beacons_.png' %(figures), dpi = 100)    
plt.show()


# ## 4. combine the two arrays into one - 0th column lenght of trajectory, 1st column lenght of straight line 

# In[16]:


resultant= (np.asarray(longs),np.asarray(straights))
np.asarray(resultant).shape


# In[17]:


def ratios (list1,list2):
    resultant= (np.asarray(list1),np.asarray(list2))
    div = []
    for i in range(len(resultant[1])):
        div.append(resultant[1][i]/resultant[0][i])
    return np.asarray(div)
div = ratios(longs,straights)


# ## 5. Plot as a into a histogram,

# In[20]:


div=np.asarray(div)
plt.hist(div[::2])
plt.hist(div[1::2])
np.mean(div[::2])
plt.axvline(div[::2].mean(), color='blue', linestyle='dashed', linewidth=1)
plt.axvline(div[1::2].mean(), color='orange', linestyle='dashed', linewidth=1)
print(div[0::2].mean())
print(div[1::2].mean())


# ## 6.Make a function out of it

# #### This function takes beaconn and position and graphs the percentage of lenght of trajecotries for a visible and invisible beacon - FOR INDIVIDUAL SESSIONS
# 
# 

# In[150]:


Day86_fs2 = pd.read_csv(root+'position 20200128-160013.txt',sep=" ", header=None)
Day86_fs1 = pd.read_csv(root+'position 20200128-151826.txt',sep=" ", header=None)

beacon_Day86_fs2 = pd.read_csv(root+'beacons 20200128-160013.txt',sep=" ", header=None)
beacon_Day86_fs1 = pd.read_csv(root+'beacons 20200128-151826.txt',sep=" ", header=None)

beacon_data = beacon_Day86_fs1
position_data = Day86_fs1

def resultant_lenght_vis_invis(position_data, beacon_data,seconds_back,name):
    """This function takes beaconn and position and graphs the percentage of lenght of trajecotries for a visible and invisible beacon - FOR INDIVIDUAL SESSIONS"""
    beacon_d = beacon_data.to_numpy()
    pos_data = position_data.to_numpy()
    beacon_d[:, 0] -= pos_data[0][0]
    pos_data[:, 0] -= pos_data[0][0]
    
    idxs = get_index_at_pos(beacon_d, pos_data)
    beacon_travel = get_positions_before(seconds_back, idxs,pos_data)
    straights=[]
    longs=[]
    for beacon in range(len(beacon_travel)):
        longs.append(calculate_Distance(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3]))
        straights.append(math.sqrt((beacon_travel[beacon][0,1] - beacon_travel[beacon][-1,1]) ** 2 + (beacon_travel[beacon][0,3] - beacon_travel[beacon][-1,3]) ** 2))
    div = ratios(longs,straights)

    plt.hist(div[::2],alpha=.5,color='cyan', edgecolor='blue',label='visible')
    plt.hist(div[1::2],alpha=.5,color='gold', edgecolor='y', label='invisible')
    blue_line = mlines.Line2D([], [], color='blue', marker='_',
                          markersize=15, label='Blue stars')
    plt.legend()
    np.mean(div[::2])
    plt.axvline(div[::2].mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(div[1::2].mean(), color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(div[::2]), color='blue', linestyle='solid', linewidth=1)
    plt.axvline(np.median(div[1::2]), color='orange', linestyle='solid', linewidth=1)
    #plt.axvline(np.std(div[::2]), color='blue', linestyle='dashdot', linewidth=1)
    #plt.axvline(np.std(div[1::2]), color='orange', linestyle='dashdot', linewidth=1)
    print(div[0::2].mean())
    print(div[1::2].mean())
    plt.title('resultant lenght ratios of %s visible and invisible %s sec'%(name,seconds_back))
    plt.savefig('%sresultant_lenght_ratios_%s_visible_invisible_%s.png' %(figures,seconds_back,name ), dpi = 100) 

    
resultant_lenght_vis_invis(position_data,beacon_data,3,'Day_86_fs_1')


# In[151]:


resultant_lenght_vis_invis(Day86_fs2 ,beacon_Day86_fs2,3,'Day86_fs2')


# In[152]:


resultant_lenght_vis_invis(Day87_fs2 ,beacon_Day87_fs2,3,'Day87_fs2')


# In[153]:


resultant_lenght_vis_invis(Day88_fs2 ,beacon_Day88_fs2,3,'Day88_fs2')


# In[154]:


resultant_lenght_vis_invis(Day88_fs1 ,beacon_Day88_fs1,3,'Day88_fs1')


# ## 7. Calculate histogram over all sessions 

# In[155]:


beacons = [beacon_Day86_fs1,beacon_Day87_fs1,beacon_Day88_fs1,beacon_Day89_fs1,beacon_Day90_fs1,beacon_Day91_fs1,beacon_Day92_fs1,beacon_Day93_fs1]
beacons2 = [beacon_Day86_fs2,beacon_Day87_fs2,beacon_Day88_fs2,beacon_Day89_fs2,beacon_Day90_fs2,beacon_Day91_fs2,beacon_Day92_fs2,beacon_Day93_fs2]
list_of_days = [Day86_fs1,Day87_fs1,Day88_fs1,Day89_fs1,Day90_fs1,Day91_fs1,Day92_fs1,Day93_fs1]
list_of_days2 = [Day86_fs2,Day87_fs2,Day88_fs2,Day89_fs2,Day90_fs2,Day91_fs2,Day92_fs2,Day93_fs2]
Day_list = list_of_days+list_of_days2
Beacon_list = beacons+beacons2
len(list_of_days)== len(beacons) 


# In[157]:


def resultant_lenght_vis_invis_all (list_of_days,beacon,seconds_back):
    div = []
    for index,(position,beaconz) in enumerate(zip (Day_list,Beacon_list)):  
        beacon_d = beaconz.to_numpy()
        pos_data = position.to_numpy()
        beacon_d[:, 0] -= pos_data[0][0]
        pos_data[:, 0] -= pos_data[0][0]
        idxs = get_index_at_pos(beacon_d, pos_data)
        beacon_travel = get_positions_before(seconds_back, idxs ,pos_data)    
        straights=[]
        longs=[]
        for beacon in range(len(beacon_travel)):
            longs.append(calculate_Distance(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3]))
            straights.append(math.sqrt((beacon_travel[beacon][0,1] - beacon_travel[beacon][-1,1]) ** 2 + (beacon_travel[beacon][0,3] - beacon_travel[beacon][-1,3]) ** 2))
        div.append(np.asarray((ratios(longs,straights))))
    return(np.asarray(div))

large_div = resultant_lenght_vis_invis_all(Day_list, Beacon_list,4)


# In[158]:


large_div[1]


# In[159]:


def histogram_ratio_all (list_of_days,beacon,seconds_back):

    large_div = resultant_lenght_vis_invis_all (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]

    for div in range(len(large_div)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div[div][::2].mean())
        large_mean_invis.append(large_div[div][1::2].mean())
        large_median_vis.append(np.median(large_div[div][::2]))
        large_median_invis.append(np.median(large_div[div][1::2]))
    vis = [item for sublist in large_div for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div for item in sublist[1::2]]
    print(ranksums(vis, invis))
    plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')
    plt.axvline((np.mean(np.asarray(large_mean_vis))-np.std(vis)), color='blue', linestyle='dashdot', linewidth=1,label='std_vis')
    plt.axvline((np.mean(np.asarray(large_mean_invis))-np.std(invis)), color='orange', linestyle='dashdot', linewidth=1,label='std_vis')

    

    plt.axvline(np.mean(np.asarray(large_mean_vis)), color='g', linestyle='dashed', linewidth=1,label='mean_vis')
    plt.axvline(np.mean(np.asarray(large_mean_invis)), color='black', linestyle='dashed', linewidth=1,label='mean_invis')
    plt.axvline(np.median(np.asarray(large_median_vis)), color='g', linestyle='solid', linewidth=1,label='median_vis')
    plt.axvline(np.median(np.asarray(large_median_invis)), color='black', linestyle='solid', linewidth=1,label='median_invis')
    plt.xlabel("ratio short/long ")
    plt.legend()
    print (seconds_back)
    plt.title('resultant lenght ratios of visible and invisible Group %s sec'% seconds_back)
    plt.savefig('%sresultant_lenght_ratios_%s_visible_invisible_all.png' %(figures,seconds_back), dpi = 200) 
    plt.show()
histogram_ratio_all (Day_list, Beacon_list , 4 )


# In[160]:


histogram_ratio_all (Day_list, Beacon_list,3)


# In[161]:


histogram_ratio_all (Day_list, Beacon_list,2)


# In[162]:


histogram_ratio_all (Day_list, Beacon_list,1)


# In[163]:


histogram_ratio_all (Day_list, Beacon_list,5)


# In[164]:


histogram_ratio_all (Day_list, Beacon_list,6)


# In[165]:


histogram_ratio_all (Day_list, Beacon_list,3.5)


# In[276]:


def scatter_ratio_all (list_of_days,beacon,seconds_back):

    large_div = resultant_lenght_vis_invis_all (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]

    for div in range(len(large_div)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div[div][::2].mean())
        large_mean_invis.append(large_div[div][1::2].mean())
        large_median_vis.append(np.median(large_div[div][::2]))
        large_median_invis.append(np.median(large_div[div][1::2]))
    vis = [item for sublist in large_div for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div for item in sublist[1::2]]
    #plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    #plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')

    print(len(vis),len(invis),)
    plt.scatter(vis[:189],invis[:189],marker="+")
    print(ranksums(vis, invis))

    plt.xlabel('visible ratio')
    plt.ylabel('invisible ratio')

    plt.axvline(np.mean(np.asarray(large_mean_vis)), color='g', linestyle='dashed', linewidth=1,label='mean_vis')
    plt.axhline(np.mean(np.asarray(large_mean_invis)), color='black', linestyle='dashed', linewidth=1,label='mean_invis')
    plt.axvline(np.median(np.asarray(large_median_vis)), color='g', linestyle='solid', linewidth=1,label='median_vis')
    plt.axhline(np.median(np.asarray(large_median_invis)), color='black', linestyle='solid', linewidth=1,label='median_invis')
    print (seconds_back)
    plt.legend()
    plt.title('resultant lenght ratios of visible and invisible Group %s sec'% seconds_back)
    plt.savefig('%sresultant_lenght_ratios_scatter_%s_visible_invisible_all.png' %(figures,seconds_back), dpi = 200) 
    plt.show()
scatter_ratio_all (Day_list, Beacon_list , 2 )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 8. Sham calculations 
# 
# 1. Create random numbers based on the lenght of the recordign and the amount of beacons. 
# 2. Use the indexes to index into the data,  
# 3. Generate the histograms and resultant lenght for that data. 
# 
# 

# In[170]:


import random as rn


# In[32]:


Day86_fs2 = pd.read_csv(root+'position 20200128-160013.txt',sep=" ", header=None)
Day86_fs1 = pd.read_csv(root+'position 20200128-151826.txt',sep=" ", header=None)

beacon_Day86_fs2 = pd.read_csv(root+'beacons 20200128-160013.txt',sep=" ", header=None)
beacon_Day86_fs1 = pd.read_csv(root+'beacons 20200128-151826.txt',sep=" ", header=None)

beacon_data = beacon_Day86_fs1
position_data = Day86_fs1

print(len(beacon_data))
print(len(position_data))

rn.randrange(0, len(position_data),len(beacon_data))
my_randoms = rn.sample(range(1, len(position_data)), len(beacon_data))

print(len(my_randoms))
print(max(my_randoms))


# ### Perhaps need to sort the random numbers... 
# 

# In[227]:


indexes = get_index_at_pos(beacon_d, pos_data)
def get_positions_before_sham(seconds_back, idxs, position_data):
    """create arrays of positions before beacon reached"""
    beacon_periods = []
    randoms = rn.sample(range(1, len(position_data)), len(idxs))
    randoms.sort()
    for beacon_idx in randoms:
        beacon_t = position_data[beacon_idx][0]
        beacon_t_before = beacon_t - seconds_back
        before_idx = np.abs(beacon_t_before - position_data[:, 0]).argmin()
        beacon_periods.append(position_data[before_idx:beacon_idx])
        
    return beacon_periods


l =get_positions_before_sham(2.2,indexes,pos_data)
l[10][10,0]


# In[228]:


def resultant_lenght_vis_invis_all_sham (list_of_days,beacon,seconds_back):
    div = []
    for index,(position,beaconz) in enumerate(zip (Day_list,Beacon_list)):  
        beacon_d = beaconz.to_numpy()
        pos_data = position.to_numpy()
        beacon_d[:, 0] -= pos_data[0][0]
        pos_data[:, 0] -= pos_data[0][0]
        idxs = get_index_at_pos(beacon_d, pos_data)
        beacon_travel = get_positions_before_sham(seconds_back, idxs ,pos_data)    
        straights=[]
        longs=[]
        for beacon in range(len(beacon_travel)):
            longs.append(calculate_Distance(beacon_travel[beacon][:,1],beacon_travel[beacon][:,3]))
            straights.append(math.sqrt((beacon_travel[beacon][0,1] - beacon_travel[beacon][-1,1]) ** 2 + (beacon_travel[beacon][0,3] - beacon_travel[beacon][-1,3]) ** 2))
        div.append(np.asarray((ratios(longs,straights))))
    return(np.asarray(div))

large_div_sham = resultant_lenght_vis_invis_all_sham(Day_list, Beacon_list,4)


# In[230]:


def histogram_ratio_all_sham  (list_of_days,beacon,seconds_back):

    large_div_sham = resultant_lenght_vis_invis_all_sham (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]

    for div in range(len(large_div)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div_sham[div][::2].mean())
        large_mean_invis.append(large_div_sham[div][1::2].mean())
        large_median_vis.append(np.median(large_div_sham[div][::2]))
        large_median_invis.append(np.median(large_div_sham[div][1::2]))
    vis = [item for sublist in large_div_sham for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div_sham for item in sublist[1::2]]

    plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')



    plt.axvline(np.mean(np.asarray(large_mean_vis)), color='g', linestyle='dashed', linewidth=1,label='mean_vis')
    plt.axvline(np.mean(np.asarray(large_mean_invis)), color='black', linestyle='dashed', linewidth=1,label='mean_invis')
    plt.axvline(np.median(np.asarray(large_median_vis)), color='g', linestyle='solid', linewidth=1,label='median_vis')
    plt.axvline(np.median(np.asarray(large_median_invis)), color='black', linestyle='solid', linewidth=1,label='median_invis')
    plt.xlabel("ratio short/long ")
    plt.legend()
    print (seconds_back)
    plt.title('resultant lenght ratios of visible and invisible Group_sham %s sec'% seconds_back)
    plt.savefig('%sresultant_lenght_ratios_%s_visible_invisible_all_sham.png' %(figures,seconds_back), dpi = 200) 
    plt.show()
histogram_ratio_all_sham (Day_list, Beacon_list , 3 )


# ### 9. Bootstrapping 

# In[231]:


def histogram_ratio_all_boot  (list_of_days,beacon,seconds_back):

    large_div_sham = resultant_lenght_vis_invis_all_sham (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]

    for div in range(len(large_div_sham)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div_sham[div][::2].mean())
        large_mean_invis.append(large_div_sham[div][1::2].mean())
        large_median_vis.append(np.median(large_div_sham[div][::2]))
        large_median_invis.append(np.median(large_div_sham[div][1::2]))
    vis = [item for sublist in large_div_sham for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div_sham for item in sublist[1::2]]

    #plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    #plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')

    #plt.legend()

    mean_vis= np.mean(np.asarray(large_mean_vis)), 
    mean_invis = np.mean(np.asarray(large_mean_invis)), 
    median_vis = np.median(np.asarray(large_median_vis)), 
    median_invis = np.median(np.asarray(large_median_invis)),
    #print (seconds_back)
    return [mean_vis,mean_invis, median_vis,median_invis]
    
histogram_ratio_all_boot (Day_list, Beacon_list , 3 )


# ## Bootstrap - calculate means and sampled data over X times also for whatever times 

# In[212]:


ave=[]
for i in range (10):
    ave.append(histogram_ratio_all_boot (Day_list, Beacon_list , 2 ))
    


# In[232]:


def strapped_means (ave):
    ave_all = []
    mean_vis_boot =[]
    mean_invis_boot=[]
    median_vis_boot=[]
    median_invis_boot=[] 
    bins=25
    for i in range(len(ave)):
        mean_vis_boot.append(ave[i][0])
        mean_invis_boot.append(ave[i][1])
        median_vis_boot.append(ave[i][2])
        median_invis_boot.append(ave[i][3])
        

    return [np.mean(mean_vis_boot), np.mean(mean_invis_boot), np.median(np.asarray(median_vis_boot)),np.median(median_invis_boot)]


# In[233]:


ave_all_boot= strapped_means(ave)


# In[234]:


ave_all_boot



# ## Function to generate stats for given seconds... 
# 

# In[236]:


def get_boot_data(seconds_back,boot_reps):
    histogram_ratio_all_boot (Day_list, Beacon_list , seconds_back )
    ave=[]
    for i in range (boot_reps):
        ave.append(histogram_ratio_all_boot (Day_list, Beacon_list , 2 ))
    ave_all_boot= strapped_means(ave)
    return ave_all_boot
    
get_boot_data(3,10)


# ### statistics on ratios of the original correctly samples data

# In[237]:


def histogram_ratio_all_nums  (list_of_days,beacon,seconds_back):

    large_div = resultant_lenght_vis_invis_all (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]

    for div in range(len(large_div)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div[div][::2].mean())
        large_mean_invis.append(large_div[div][1::2].mean())
        large_median_vis.append(np.median(large_div[div][::2]))
        large_median_invis.append(np.median(large_div[div][1::2]))
    vis = [item for sublist in large_div for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div for item in sublist[1::2]]

    #plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    #plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')

    #plt.legend()

    mean_vis= np.mean(np.asarray(large_mean_vis)), 
    mean_invis = np.mean(np.asarray(large_mean_invis)), 
    median_vis = np.median(np.asarray(large_median_vis)), 
    median_invis = np.median(np.asarray(large_median_invis)),
    #print (seconds_back)
    return [mean_vis,mean_invis, median_vis,median_invis]
    
ave_all = histogram_ratio_all_nums (Day_list, Beacon_list , 3 )


# In[238]:


ave_all


# ## 10. Graph together with bootstrapped data: 

# In[241]:


def histogram_ratio_with_sham (list_of_days,beacon,seconds_back,boot_reps):

    large_div = resultant_lenght_vis_invis_all (list_of_days,beacon,seconds_back)


    large_mean_vis=[]
    large_median_vis=[]
    large_mean_invis=[]
    large_median_invis=[]
    
    ave_all_boot = get_boot_data(seconds_back,boot_reps)
    
    for div in range(len(large_div)):

        #within group stats - not pooled 
        large_mean_vis.append(large_div[div][::2].mean())
        large_mean_invis.append(large_div[div][1::2].mean())
        large_median_vis.append(np.median(large_div[div][::2]))
        large_median_invis.append(np.median(large_div[div][1::2]))
    vis = [item for sublist in large_div for item in sublist[::2]]  #cool list feature - flatening lists
    invis = [item for sublist in large_div for item in sublist[1::2]]
    print(ranksums(vis, invis))
    plt.hist(vis,alpha=.5,color='g', edgecolor='seagreen',label='visible')
    plt.hist(invis,alpha=.5,color='lightgrey', edgecolor='silver',label='invisible')
    plt.axvline((np.median(np.asarray(large_median_vis))-np.std(vis)), color='blue', linestyle='dashdot', linewidth=1,label='std_vis')
    plt.axvline((np.median(np.asarray(large_median_invis))-np.std(invis)), color='orange', linestyle='dashdot', linewidth=1,label='std_invis')
    plt.axvline(ave_all_boot[2], color='purple', linestyle='dashed', linewidth=1,label='sham_med_vis')
    plt.axvline(ave_all_boot[3], color='pink', linestyle='dashed', linewidth=1,label='sham_med_invis')
    #plt.axvline(np.mean(np.asarray(large_mean_vis)), color='g', linestyle='dashed', linewidth=1)
    #plt.axvline(np.mean(np.asarray(large_mean_invis)), color='black', linestyle='dashed', linewidth=1)
    plt.axvline(np.median(np.asarray(large_median_vis)), color='g', linestyle='solid', linewidth=1,label='med_vis')
    plt.axvline(np.median(np.asarray(large_median_invis)), color='black', linestyle='solid', linewidth=1,label='med_invis')
    plt.legend()
    plt.xlabel("ratio short/long ")
    print (seconds_back)
    plt.title('resultant lenght ratios of visible and invisible Group_with_sham %s sec'% seconds_back)
    plt.savefig('%sresultant_lenght_ratios_%s_visible_invisible_all_with_sham.png' %(figures,seconds_back), dpi = 200) 
    plt.show()
histogram_ratio_with_sham (Day_list, Beacon_list , 3,10 )


# In[243]:


histogram_ratio_with_sham (Day_list, Beacon_list , 2,10 )


# In[189]:


histogram_ratio_with_sham (Day_list, Beacon_list , 1,20 )


# In[190]:


histogram_ratio_with_sham (Day_list, Beacon_list , 4,20 )


# ## Conclusion: 
#     Computign the ratio differences showed no significant differecnes between the lenght ratios of resultant lenght between visible and invisble beacon condition. There was a slight preference at 2 and 3 seconds before the beacon, and when calculatign sham it showed that those ratios are onaverage much smaller than the given ratio from the trials, but not significatonly so.  

# ####        Note, need to always subtract STD from the mean 
#     

# ## 11. Sliding median window. 
# 1. calculate median and meand in an array for .1 sec each 
# 2. calculate simliary for sham condition - 20 repetitions or so 
# 3. Plot where x axis are the time points and y woudl be medians of ratios - 4 lines vis, and invis for sham and normal 
# 
# 

# In[268]:


np.seterr('warn')
run_ave=[]
for i in range(1,200,1):
    run_ave.append(histogram_ratio_all_nums (Day_list, Beacon_list ,i/10 ))


# In[269]:


run_ave_sham=[]
for i in range(1,200,1):
    run_ave_sham.append(histogram_ratio_all_boot (Day_list, Beacon_list ,i/10 ))


# ### [mean_vis,mean_invis, median_vis,median_invis]

# In[272]:


sns.set()
r= range(1,200,1)


# In[273]:


r1=[]
medians_vis=[]
medians_invis=[]
medians_vis_sham=[]
medians_invis_sham=[]
for i in range (198):
    r1.append(r[i]/10)
    medians_vis.append(run_ave[i][2])
    medians_invis.append(run_ave[i][3])
    medians_vis_sham.append(run_ave_sham[i][2])
    medians_invis_sham.append(run_ave_sham[i][3])
    

plt.plot(r1,medians_vis,label='median_vis')
plt.plot(r1,medians_invis, label='median_vis')
plt.plot(r1,medians_vis_sham, label='sham_median_vis')
plt.plot(r1,medians_invis_sham, label='sham_median_vis')
plt.xlabel('time(s)')
plt.ylabel('resultant lenght ratio medians')
plt.title('resultant lenght ratios running medians with sham')
plt.legend()
plt.savefig('%sresultant_lenght_ratios_running_medians%s.png' %(figures,i), dpi = 200) 


# In[275]:


r1=[]
means_vis=[]
means_invis=[]
means_vis_sham=[]
means_invis_sham=[]
for i in range (199):
    r1.append(r[i]/10)
    means_vis.append(run_ave[i][0])
    means_invis.append(run_ave[i][1])
    means_vis_sham.append(run_ave_sham[i][0])
    means_invis_sham.append(run_ave_sham[i][1])
    

plt.plot(r1,means_vis,label='mean_vis')
plt.plot(r1,means_invis,label='mean_invis')
plt.plot(r1,means_vis_sham,label='sham_mean_invis')
plt.plot(r1,means_invis_sham,label='sham_mean_invis')
plt.xlabel('time(s)')
plt.ylabel('resultant lenght ratio means')
plt.title('resultant lenght ratios running means with sham')
plt.legend()
plt.savefig('%sresultant_lenght_ratios_running_means%s.png' %(figures,i), dpi = 200) 


# #### DEBUGGING   - Problems 
# 1. Trajecotry check - maybe taking 5 seconds instead of 3 as I think due to the pyhton program which has some frame rate at 
# 50hz not always at 100 
# 2. for some reason indexign spits out at 2.2 due to the wrong shape (3,221) where it somehow rounds and takes an extra index. - try to fix below, but still does not work  - Kind of fixed manually 
# 3. Gettign NAN on the bootstrap data - due to division by zero?  - maybe need ot normalize time values due to the didison of small numbers where numpy fails 
# 

# In[198]:


time_list=[]
x_list=[]
time_list.append(position_data[0][(int(418-(2.2*100))):int(417)])
x_list.append(position_data[1][int(418-(2.2*100)):int(418)])


print(len(time_list[0]))
print(len(x_list[0]))
print ((int(418-(2.2*100)-418))*-1)
len(time_list[0]) == (int(418-(2.2*100)-418))*-1


# In[209]:


def position_before_beacon_trigger_beacon_array(seconds_back, beacon_data, position_data):
    """Take beacon data and returns XY and Time array defined in seconds before beacon """
    x_list=[]
    y_list=[]
    time_list=[] 
    num=0
    beacon_travel2=[]
    index, enum  = get_index(beacon_data, position_data)
    for i in index:
        if i <= seconds_back*100:
            i = int(seconds_back*100+1)
            print("small")
        x_list.append(position_data[1][int(i-(seconds_back*100)):int(i)])
        y_list.append(position_data[3][int(i-(seconds_back*100)):int(i)])
        time_list.append((position_data[0][int(i-(seconds_back*100)):int(i)]-position_data[0][0]))
        if (len(time_list[0]) < (int(i-(seconds_back*100)-i))*-1):
            print('érror')
            #print(len(time_list[0]))
            #print(i)
            time_list[0]= time_list[0][0:int((seconds_back*100))]
            y_list[0]= y_list[0][0:int((seconds_back*100))]
            x_list[0]= x_list[0][0:int((seconds_back*100))]
        #print(i)
        #if (len(time_list[0]) != (int(i-(seconds_back*100)-i))*-1):
         #   print('big')
          #  time_list[0].append(time_list[0][-1]) #np.asarray(time_list[0][0:int((seconds_back*100))]).mean
            
        #print(len(time_list[0]))
        #print(len(x_list[0]))
        k= np.asarray((time_list[0],x_list[0],y_list[0],))
        #print(k.shape)
        num=num+1
        beacon_travel2.append(k)
        x_list=[]
        y_list=[]
        time_list=[]
    return np.asarray(beacon_travel2)
beacon_travel= position_before_beacon_trigger_beacon_array(2.2,beacon_data, position_data)


# In[ ]:





# In[ ]:




