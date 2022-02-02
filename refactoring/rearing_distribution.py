import numpy as np
import base
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.image as mpimg

## Global variables for image normalization

cut = 0 # keeping the cut where rectangle of arena ends
X_cut_min = -.59
Y_cut_max = 1.61
X_cut_max = .12
Y_cut_min = .00

x_max, x_min = 0.12, -0.59
x_offset = x_max - (x_max - x_min)/2
y_max, y_min = 1.61,  0
y_offset = y_max - (y_max - y_min)/2


# to center the zero in... 
xcut_offset=-.24
ycut_offset=-.8

cut=0

def make_image(center=(.1,-.4),dpi=500,X_cut_min = -.59 -xcut_offset,Y_cut_max = 1.61
               + ycut_offset,X_cut_max = .12-xcut_offset,Y_cut_min = .00 +ycut_offset,bands=23 ):
    """make visual count it by area then have hist values for normalization wih movement data
        to be exported and then can be counted
    
    PARAMS
    ------------
    center : tuple 
        where beacon is 
    dpi : int 
        dots per inch - resolution - if changed can mess up pixel count
    X_cut,Y_cut : int
        points of rectagle, same as used for cutting of rears - floor of arena  
        
    bands : int 
        amount of circles fittign inthe rectangle - max is 23 
        
    Returns
    ------------
    Histogram and appropriate bins made by the histogram
    Used for area estimation later on

    """
    
    fig, ax1 = plt.subplots(1, 1, sharex=True,dpi=dpi,)
    fig.patch.set_visible(False)
    rectangle = patches.Rectangle((X_cut_min,Y_cut_min), (abs(X_cut_min)+abs(X_cut_max)),abs(Y_cut_min)+abs(Y_cut_max) , color="white")
    ax1.add_patch(rectangle)
    #plt.plot(center[0],center[1], "ro")
    color = np.linspace(0,.99,bands+1)
    for i in reversed(range(bands)):
        c=color[i]
        patch = patches.Circle((center[0],center[1]), radius=.075*i,color=str(c))
        ax1.add_patch(patch)
        patch.set_clip_path(rectangle)                  
    ax1.axis("equal")
    ax1.axis("off")
    fig.savefig('norm_graph.png', dpi=dpi, transparent=True)
    img= Image.frombytes('RGB',fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    image_array = np.asarray(img)
    hist, bins = np.histogram(image_array,bins=bands,range=(0,249))
    #plt.show()
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    #plt.bar(center, hist, align='center', width=width)
    
    return hist,bins 

def get_normalized_rears(session_data, session_rearings, session_no):
    trial_rears_bin_counts = []
    trial_hist = []
    trial_bins = []
    trial_visibility = []
    for beacon_pos, rears, visibility in zip(session_data.beacon_list[session_no], session_rearings[session_no], session_data.trial_visible[session_no]):
        trial_visibility.append(visibility)
        hist, bins = make_image(beacon_pos)
        bins_idx = np.digitize(rears*100,bins)
        rears_bin_counts=np.zeros_like(bins)
        for i in range(len(bins)):
            rears_bin_counts[i]=np.sum(bins_idx == i)
        
        trial_rears_bin_counts.append(rears_bin_counts)
        trial_hist.append(hist)
        trial_bins.append(bins)

    return trial_rears_bin_counts, trial_hist, trial_bins, trial_visibility

