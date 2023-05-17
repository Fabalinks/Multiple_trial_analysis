Rearing detection tried by Miao Wang

Recent update:
1. labled data of where the flip happen: data/processed_data/FS11/distance, 2022-03-30.npy works well based on the plot of pitch vs height. The performance varies from session to session. Only update two files due to size limitation.

Load the data by 
all_data= np.load('2022-03-30.npy',allow_pickle=True).item()
flip_label = all_data['flip_idx'] 
markers = all_data['markers']

2. demo_fix_jumps.ipynb shows how to generate processed data, but I did not upload the raw data due to size limitation.

Already finished:

    # 1 baseline method: use height, z speed, xy plannar speed to determine the rearing. 
        Advantages: 
            1. Extend the rearing period detected by set a height threshold.
            2. Easy
        Shortcoming:
            1. Can not well detect the begin and end of rearing
            2. not solid statistical-based
        

        
    
    # 2. PCA methods: 
        1. use pca or svd to detect rearing
        2. Try to figure out what each PC is capturing by plot the joint distribution of reconstructed data
        
        Problem:
            1. The begin of the rearing is not precise(as shown in resulst/PCA_method/FS10), depend on the value of tau
    


    # 3. ICA methods:
        1. Use ICA to decompose the features and detect rearing

        Problem:
            1. I tried different methods to preprocess the data (global centering: subtract mean in each example, local centering: subtract mean in each dimension(the column)), the resutls showed that for global centering, the results could noe converge.

    
    # 4. Dynamic of different features before rearing:
        See how features value evolue as time approach rearing time by ploting joint distribution along time window.

Not finished yet:

    # 1. Distinguish supported rearing and unsupported rearing:
        Problem: what is the boarder on xy ?
        From Fabian's code:
            X_cut_min = -.6
            Y_cut_max = 1.5
            X_cut_max = .05
            Y_cut_min = .08

 