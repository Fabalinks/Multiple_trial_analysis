Rearing detection tried by Miao Wang


Already finished:
    baseline method: use height, z speed, xy plannar speed to determine the rearing. 
        Advantages: 
            1. Extend the rearing period detected by set a height threshold.
            2. Easy
        Shortcoming:
            1. Can not well detect the begin and end of rearing
            2. not solid statistical-based
        
        Plan to do:
            Use change point detection to determine the begin and end of rearing
        
    
    PCA methods: use pca or svd to detect rearing
        Finished part:
            1. generate design matrix for pca, the reuslt seem to be ok, big explained ratio for the first pc.
            2. get some relationship of different features(height, pitch, zspeed, xy speed) from first PC eigenvector
        
        Need to do:
            Apply this relationship to determine rearing.

