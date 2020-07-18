import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    #os.path.basename()返回path最后的文件名
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    #ldx是打乱的列表，包含40个次序
    return data[idx, ...], labels[idx], idx
    #...省略所有：

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    #(32,1024,3),对每个批次的数据进行旋转
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        #(3,3)
        shape_pc = batch_data[k, ...]#(k,1024,3)
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)#断言
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    #jittered_data=np.clip(0.01*randn(B, N, C), -0.05, 0.05)

# ''' >>> a = np.arange(10)
#     >>> np.clip(a, 1, 8)
#     array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8]) # a被限制在1-8之间
#     >>> a
#     array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # 没改变a的原值
#
#     >>> np.clip(a, 3, 6, out=a) # 修剪后的数组存入到a中
#     array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])'''

    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    '''Input:txt
        Out:[]
    '''
    return [line.rstrip() for line in open(list_filename)]
    #rstrip() 删除 string 字符串末尾的指定字符（默认为空格）

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
    #(2048, 2048, 3) (2048, 1)

def loadDataFile(filename):
    return load_h5(filename)



def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
