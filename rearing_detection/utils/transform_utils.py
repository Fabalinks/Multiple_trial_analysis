import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation 

def get_quaternion(before_coord,after_coord):
    '''
    Calculate the quaternion for rotation of the coordinate system. 
    
    before_coord: coordinates of marker before rotation, np.array in shape [3,3], where [0,:] is the coordinate of the first dot before rotation
    after_coord: coordinates of marker after the rotation, np.array in shape [3,3], where [0,:] is the coordinate of the first dot after rotation. 
    
    Note: Do NOT consider translation here, so to generate after_coord, we need minus the translation distance if there is translation
    Return:
        quat: np.array in shape [4,], quaternion result [qw, qx, qy, qz] 
        
    '''
    matchlist=range(len(before_coord))
    M=np.matrix([[0,0,0],[0,0,0],[0,0,0]])

    for i,coord1 in enumerate(before_coord):
        x=np.matrix(np.outer(coord1,after_coord[matchlist[i]]))
        M=M+x

    N11=float(M[0][:,0]+M[1][:,1]+M[2][:,2])
    N22=float(M[0][:,0]-M[1][:,1]-M[2][:,2])
    N33=float(-M[0][:,0]+M[1][:,1]-M[2][:,2])
    N44=float(-M[0][:,0]-M[1][:,1]+M[2][:,2])
    N12=float(M[1][:,2]-M[2][:,1])
    N13=float(M[2][:,0]-M[0][:,2])
    N14=float(M[0][:,1]-M[1][:,0])
    N21=float(N12)
    N23=float(M[0][:,1]+M[1][:,0])
    N24=float(M[2][:,0]+M[0][:,2])
    N31=float(N13)
    N32=float(N23)
    N34=float(M[1][:,2]+M[2][:,1])
    N41=float(N14)
    N42=float(N24)
    N43=float(N34)

    N=np.matrix([[N11,N12,N13,N14],\
              [N21,N22,N23,N24],\
              [N31,N32,N33,N34],\
              [N41,N42,N43,N44]])
    values,vectors=np.linalg.eig(N)
    w=list(values)
    mw=max(w)
    quat= vectors[:,w.index(mw)]
    quat=np.array(quat).reshape(-1,).tolist()
    return quat


def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (x,y,z,w)
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    sample_num = Q.shape[0]
    rot_matrixs = np.zeros((sample_num, 3,3))
    
    q0 = Q[:,3] # w for all sample
    q1 = Q[:,0] # x for all sample
    q2 = Q[:,1] # y for all sample
    q3 = Q[:,2] # z for all sample
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # nx3x3 rotation matrix
    rot_matrixs[:,0,0] = r00
    rot_matrixs[:,0,1] = r01
    rot_matrixs[:,0,2] = r02
    rot_matrixs[:,1,0] = r10
    rot_matrixs[:,1,1] = r11
    rot_matrixs[:,1,2] = r12
    rot_matrixs[:,2,0] = r20
    rot_matrixs[:,2,1] = r21
    rot_matrixs[:,2,2] = r22
                            
    return rot_matrixs

def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) angle to euclidean (x, y, z) angles, in degrees.
    Equation from http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/"""
    
    heading = np.arctan2(2*qy*qw-2*qx*qz , 1 - 2*qy**2 - 2*qz**2)
    attitude = np.arcsin(2*qx*qy + 2*qz*qw)
    bank = np.arctan2(2*qx*qw-2*qy*qz , 1 - 2*qx**2 - 2*qz**2)
    
    return np.array([np.degrees(angle) for angle in [attitude, heading, bank]]) #[roll, yaw, pitch]


def make_uit_ax(vec):
    '''
    unify the vector by divided by the norm
    vec: vector in np.array
    '''
    unit_vec = vec/np.linalg.norm(vec)
    return unit_vec


def determine_self_coor(origin, self_ax, camera_coor):
    '''
    camera_coor: usually in shape [4,3] 
    origin: loc of the orginal dot
    '''
    relative_ax = camera_coor - origin # for each marker, get the vector from origin to the marker
    self_coor = relative_ax @ self_ax.T
    return self_coor

def calculate_angle_with_qaut(begin, end, markers, raw_data, self_coor, nan_idx):
    '''
    There are four markers of the track. Treat the center-middle marker as the origin point in mouse's self coordinate system, 
    the axis of the system: 
        x axis: parallel to the line determined by two ears.
        y axis: vertical to the horizontal, pointing up
        z axis: midline of the mouse, from nose to tail direction
        
    Calculated rotation euler angle of mouse(or to say the self coordinate system) in extrinsic x,z,y sequence. 
    
    Parameters:
        begin, end: index to determine the subgroup of time series to be calculated
        markers: np.array, whole set of four markers coordinate in time series, in shape [N, 4, 3], where N is the sample in time series, 
                [0,0,:] is the coordinates of the first marker in first frame, [:,-1,:] is the center-middle marker coords
        raw_data: pd.Dataframe generated from the raw csv file
        self_coor: np.array in shape [4,3], coordinate of four markers in self coordinate system [-1,:] is [0,0,0] (the origin)
        nan_idx: list of index to indicate nan data    
    Return:
        model_angles: angle calculated from coordinated of markers in time series
        real_angles: angle calcualted with the quaternion provided by raw file
    
        in mose cases, the resutls of model_angles and real_angles are the same, but if there is filp of the markers, model_angles will have the correct version
    
    '''
    model_angles = []
    real_angles = []
    
    idxs = range(begin,end)
    
    # to calculate the translated markers
    offset_loc = np.zeros_like(markers)
    for i in range(0,4):
        offset_loc[:,i,:] = markers[:,-1,:]

    translated_markers  = markers - offset_loc # coordinates of markers after minusing the translation

    for i,idx in tqdm(enumerate(idxs)):
        if nan_idx[idx] == True:
            real_angle= quaternion_to_euler(*raw_data.iloc[idx,34:38])
            real_angles.append(real_angle)
            model_angles.append(np.array([np.nan,np.nan, np.nan]))
            continue
        else:
            quant = np.array(get_quaternion(self_coor[0:-1],translated_markers[idx,0:-1])) # [qw, qx, qy, qz]
            reorder_quant = quant[[1,2,3,0]] #  [qx,qy,qz,qw]
            model_r = Rotation.from_quat(reorder_quant) # rotation matrix 
            model_angle = model_r.as_euler('xzy',degrees=True) # angle calculated from coordinated of markers
            real_angle = quaternion_to_euler(*raw_data.iloc[idx,34:38]) # angle calcualted with the quaternion provided by raw file
            
            
            # to record the results
            real_angles.append(real_angle)
            model_angles.append(model_angle)

    model_angles = np.array(model_angles)
    real_angles = np.array(real_angles)

    return model_angles, real_angles
        

    
def generate_marker_loc(row_id,raw_data):
    '''
    return np.array of the marker coordinate from dataframe
    '''
    offset_col = 42
    marker_ids = range(0,4)
    if len(row_id)<=1:
        markers = np.zeros((4,3))
        for i, marker_id in enumerate(marker_ids):
            markers[i,:] =   raw_data.iloc[row_id, int(marker_id*4 + offset_col):int(marker_id*4 + offset_col + 3)].values
    else:
        markers = np.zeros((len(row_id),4,3))
        for i, marker_id in enumerate(marker_ids):
            markers[:,i,:] =   raw_data.iloc[row_id, int(marker_id*4 + offset_col):int(marker_id*4 + offset_col + 3)].values
    
    return markers