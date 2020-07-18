import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #当前文件所在文件夹
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider    #数据集
import tf_util

#命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step  #每隔多少step更新一次learning rate的值
DECAY_RATE = FLAGS.decay_rate  #学习率衰减率

MODEL = importlib.import_module(FLAGS.model) # 导入模型：pointnet_cls
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')    #模型文件=pointnet_cls.py

LOG_DIR = FLAGS.log_dir       # 创建日志目录
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

os.system('copy %s %s' % (MODEL_FILE, LOG_DIR)) # back-up of model def，备份
os.system('copy train.py %s' % (LOG_DIR)) # bkp of train procedure
# os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
# os.system('cp train.py %s' % (LOG_DIR))

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w') #写入日志文件
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5                 #初始的bn-decay
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)#20000
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()   #获取主机名

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# ['data/modelnet40_ply_hdf5_2048/ply_data_train0.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train1.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train2.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train3.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_train4.h5']
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
# ['data/modelnet40_ply_hdf5_2048/ply_data_test0.h5',
# 'data/modelnet40_ply_hdf5_2048/ply_data_test1.h5']

def log_string(out_str):    #在日志文件中写入内容
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()        #清空缓存区
    print(out_str)

#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
#衰减后的学习率=初始学习率*衰减率^（全局步数/衰减步数）
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.基础学习率   0.001
                        batch * BATCH_SIZE,  # Current index into the dataset.  当前数据在数据集中的总次序
                        DECAY_STEP,          # Decay step.  衰减步数    200000
                        DECAY_RATE,          # Decay rate.  衰减率   0.7
                        staircase=True)      #阶梯形式，默认值为False,当为True时，（global_step/decay_steps）则被转化为整数
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

# BN_INIT_DECAY = 0.5
# BN_DECAY_DECAY_RATE = 0.5
# BN_DECAY_DECAY_STEP = float(DECAY_STEP)   20000
# BN_DECAY_CLIP = 0.99
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,            #0.5
                      batch*BATCH_SIZE,         #
                      BN_DECAY_DECAY_STEP,      #20000
                      BN_DECAY_DECAY_RATE,      #0.5
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)#  min(0.99,1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):    #default=0
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT) #MODEL=pointnet_cls.py
            # return tf.placeholder()
            # (32,1024,3) (32)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)   #Tensor("Placeholder:0", shape=(), dtype=bool)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)     #显示标量信息

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            #(32, 1024, 3)->(32,40)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)#降维求和
            #tf.cast()类型转换
            #准确率=sum(正确数/批数量32)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            #global_step：梯度下降一次自动加1，一般用于记录迭代优化的次数，主要用于参数输出和保存
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()  #max_to_keep=5
            # max_to_keep: 表明保存的最大checkpoint 文件数。当一个新文件创建的时候，旧文件就会被删掉。
            # 如果值为None或0，表示保存所有的checkpoint 文件。默认值为5（也就是说，保存最近的5个checkpoint文件）。
            # keep_checkpoint_every_n_hour:例如，设置 keep_checkpoint_every_n_hour=2
            # 确保没训练2个小时保存一个checkpoint 文件。默认值10000小时无法看到特征。

            #saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
            #(sess, path, global_step )
            # 在saver实例每次调用save方法时，都会创建三个数据文件和一个检查点（checkpoint）文件，
            # 权重等参数被以字典的形式保存到.ckpt.data中，
            # 图和元数据被保存到.ckpt.meta中，可以被tf.train.import_meta_graph加载到当前默认的图
            # .ckpt - index，应该是内部需要的某种索引来正确映射前两个文件；


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #刚一开始分配少量的GPU容量，然后按需慢慢的增加
        config.allow_soft_placement = True  #如果你指定的设备不存在，允许TF自动分配设备
        config.log_device_placement = False #是否打印设备分配日志
        sess = tf.Session(config=config)


        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        #保存网络结构图

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               #is_training_pl = tf.placeholder(tf.bool, shape=())
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


#训练epoch
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files打乱训练集
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    # [1, 3, 0, 2, 4]

    for fn in range(len(TRAIN_FILES)):  #for fn in range(5):
        log_string('----训练数据' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        # (2048, 2048, 3) (2048, 1) 打乱顺序
        # 'data/modelnet40_ply_hdf5_2048/ply_data_train3.h5'
        current_data = current_data[:,0:NUM_POINT,:]
        # (2048, 0:1024, 3)
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label) #去掉为1的维度
        
        file_size = current_data.shape[0] # 2048个点云图像
        num_batches = file_size // BATCH_SIZE     #批次数2048/32=64
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):            #对于每一批次
            start_idx = batch_idx * BATCH_SIZE          #开始数据序号
            end_idx = (batch_idx+1) * BATCH_SIZE        #结束数据序号
            
            # Augment batched point clouds by rotation and jittering
            #通过旋转和振动增加点云数据
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            # (0:32, 1024, 3)
            jittered_data = provider.jitter_point_cloud(rotated_data)


            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}


            # merged = tf.summary.merge_all()
            #train_op = optimizer.minimize(loss, global_step=batch)
            #pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)

            # ops = {'pointclouds_pl': pointclouds_pl,
            #        'labels_pl': labels_pl,
            #        'is_training_pl': is_training_pl,
            #        'pred': pred,
            #        'loss': loss,
            #        'train_op': train_op,
            #        'merged': merged,
            #        'step': batch}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

            #train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),sess.graph)
            train_writer.add_summary(summary, step) #保存的计算图添加2个属性

            pred_val = np.argmax(pred_val, 1)  #显示每行最大元素的序号，即预测结果
            correct = np.sum(pred_val == current_label[start_idx:end_idx]) # 一个批次内的正确个数
            total_correct += correct  # 所有批次的正确个数
            total_seen += BATCH_SIZE  #总训练数据个数
            loss_sum += loss_val

        #训练完所有batch后：
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        #每一批次平均损失=总损失/批次数
        log_string('accuracy: %f' % (total_correct / float(total_seen)))
        #正确率=总正确次数/总数据个数

#验证一次epoch
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]   # 40个0的列表
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):  #对于每个测试文件0, 1
        log_string('----测试数据' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        # (2048, 2048, 3) (2048, 1)
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            # 一个batch内的预测结果32个
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)

            for i in range(start_idx, end_idx):
            # for i in range(0, 32):
                l = current_label[i]  # batch内第i个点云图像的标签
                total_seen_class[l] += 1   #出现过的这种类别数+1，算出这种类别出现的次数
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                #这种类别的正确次数          (bool自动转换为数字）

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' %
               (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
               #类别平均正确率=（这种类别正确的次数/这种类别出现次数）


if __name__ == "__main__":
    #如果模块是被直接运行的，则代码块被运行，
    #如果模块是被导入的，则代码块不被运行
    train()
    LOG_FOUT.close()
