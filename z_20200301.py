# import tensorflow as tf
import numpy as np
# from pyforest import *
import os
import h5py

# a = [[3, 5], [2, 3]]
# for i in range(10):
#     for j in range(10):
#         print(str(i) + str(j))
# q=np.random.uniform(2, 5, 7)
# for i in q:
#     print(int(i), end='')
# list_filename=r'data/modelnet40_ply_hdf5_2048/test_files.txt'
# def getDataFiles(list_filename):
#     '''Input:txt
#         Out:[]
#     '''
#     print([line.rstrip() for line in open(list_filename)])
# getDataFiles(list_filename)

# h5_filename=r'data/modelnet40_ply_hdf5_2048/ply_data_test0.h5'
# def load_h5(h5_filename):
#     f = h5py.File(h5_filename)
#     data = f['data'][:]
#     label = f['label'][:]
#     for i in label:
#         if i > 39:
#             print(i)
#     return (data, label)
# load_h5(h5_filename)

# TRAIN_FILES=['data/modelnet40_ply_hdf5_2048/ply_data_train0.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train1.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train2.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train3.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train4.h5']
# # print(len(TRAIN_FILES))
#
# train_file_idxs=np.arange(0, 5)
# np.random.shuffle(train_file_idxs)
# for fn in range(5):
#     print(TRAIN_FILES[train_file_idxs[fn]])


# ops = {'pointclouds_pl': 'pointclouds_pl',
#        'labels_pl': 'labels_pl',
#        'is_training_pl': 'is_training_pl',
#        'pred': 'pred',
#        'loss': 'loss',
#        'train_op': 'train_op',
#        'merged': 'merged',
#        'step': 'batch'}
# feed_dict = {ops['pointclouds_pl']: 'jittered_data',
#              ops['labels_pl']: 'current_label[start_idx:end_idx]',
#              ops['is_training_pl']: 'is_training', }
#
# print(ops['pointclouds_pl'])
# print(feed_dict[ops['pointclouds_pl']])

# total_seen_class = [0 for _ in range(10)]
# print(total_seen_class)
# print("%05d" % 5)

# current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
# a='add'
# def add(x, y):
#     print('和为：%d' % (x + y))
# # if __name__ == "__main__":
for i in range(9999999):
    print(i)



