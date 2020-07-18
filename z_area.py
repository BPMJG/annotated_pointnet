# from mine import pi
#
# def ma0n():
#     print(pi)
#
# ma0n()
# print('%d+%d'%(3,5))

# import time
# # import sys
# #
# # for i in range(5):
# #     print(i)
# #     # sys.stdout.flush()
# #     time.sleep(1)
import numpy as np
# a=np.array([[1,2,3],
#            [1,5,9]])
# print(np.argmax(a,0))

# print([5 for _ in range(10)])
# # print(a.get_shape)
# print(3/2)
# print(3//2)
# ops = {'pointclouds_pl': 'pointclouds_pl',
#                    'labels_pl': 'labels_pl',
#                    'is_training_pl': 'is_training_pl',
#                    'pred': 'pred',
#                    'loss': 'loss',
#                    'train_op': 'train_op',
#                    'merged': 'merged',
#                    'step': 'batch'}
# feed_dict = {ops['pointclouds_pl']: 'jittered_data',
#                          ops['labels_pl']: 'current_label[start_idx:end_idx]',
#                          ops['is_training_pl']: 'is_training',}
# print(feed_dict)

import h5py
# h5_filename=r'C:\Users\11041\Desktop\研\代码\pointnet-master\part_seg\hdf5_data\ply_data_test0.h5'
#
# f = h5py.File(h5_filename)
# data = f['data'][:]
# label = f['label'][:]
# print(f.items())

import h5py

# x = np.arange(5)

# with h5py.File('test.h5','w') as f:
# f=h5py.File('test.h5','w')
# f.create_dataset('test_numpy',data=x)
#
# group1 = f.create_group('group1')
# group1.create_dataset('test_numpy1',data=x)
#
# group2 = group1.create_group('group2')
# group2.create_dataset('test_numpy2',data=x)
#
# a=f['test_numpy']
# b=f['group1']
# c=b['test_numpy1']
# d=b['group2']
#
# one=d['test_numpy2'][:]
# one[3]=0
# ccc=list(range(5))
# dd=sorted(a,reverse=True)

# # print(ccc)
# # print(one)
# dict={'num':(2,4,5)}
# print(dict['num'][:])
# def read_data(filename):
#     with h5py.File(filename,'r') as f:
#
#         def print_name(name):
#             print(name)
#         f.visit(print_name)
#         print('---------------------------------------')
#         subgroup = f['subgroup']
#         print(subgroup.keys())
#         print('---------------------------------------')
#         dset = f['test_numpy']
#         print(dset)
#         print(dset.name)
#         print(dset.shape)
#         print(dset.dtype)
#         print(dset[:])
#         print('---------------------------------------')
#
# read_data('test.h5')
# import tensorflow as tf
# a=np.array([[3,6,9,12,12,435,144,399],
#             [3,233,2344,12,8,1,99999,3]])
#
# plt.imshow(a)
# plt.show()
# print(a)
# x=np.random.rand(100)
# y=0.2*x+0.5
#
# Weights=tf.Variable(tf.random_uniform([1],-1.,1.,))
# bias=tf.Variable(tf.zeros([1]))
#
# y_pre=x*Weights+bias
# loss=tf.reduce_mean(tf.square(y-y_pre))
# opt=tf.train.GradientDescentOptimizer(0.1)
# train=opt.minimize(loss)
# init=tf.global_variables_initializer()
#
# sess=tf.Session()
# sess.run(init)
# for _ in range(300):
#     sess.run(train)
#     if _%15==0:
#         print(_,sess.run(Weights),sess.run(bias))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layers(input,in_size,outsize,activation_function=None):
  layer_name='n_layer'
  with tf.name_scope('layer1'):
      with tf.name_scope('Weights'):
        Weights=tf.Variable(tf.random_normal([in_size,outsize]),name='w')
        tf.summary.histogram(layer_name+'/weights',Weights)
      with tf.name_scope('bias'):
        bias=tf.Variable(tf.zeros([1,outsize])+0.01,name='b')
        tf.summary.histogram(layer_name+'/weights',Weights)

      with tf.name_scope('wxb'):
        Wx_plus_b=tf.matmul(input,Weights)+bias

  if activation_function is None:
    outputs=Wx_plus_b
    tf.summary.histogram(layer_name + '/outputs', outputs)
  else:
    outputs=activation_function(Wx_plus_b)

  return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')


l1=add_layers(xs,1,10,activation_function=tf.nn.relu)
l2=add_layers(l1,10,1,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-l2),
                   reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
init=tf.global_variables_initializer()

sess=tf.Session()
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter(r'C:\Users\11041\Desktop\log',sess.graph)

sess.run(init)

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1,)#添加子图,行数列数位置
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()

for _ in range(3000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if _%50==0:
        result=sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,_)
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        # prediction_value=sess.run(l2,
        #         feed_dict={xs:x_data})
        # lines=ax.plot(x_data,prediction_value,'r-',lw=5)
        # # ax.remove(lines[0])
        # plt.pause(0.1)



# print(tf.square(y_data-l2).shape)

# print(x_data)