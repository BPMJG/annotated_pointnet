import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):      #(32,1024)
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl
    # (32,1024,3) (32)

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    #T-NET(1),input_transform_net
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
        # (32,3,3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    #(32,1024,3)*(32,3,3)->(32,1024,3)论文第2格

    input_image = tf.expand_dims(point_cloud_transformed, -1)
    #(32,1024,3,1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    #(32,1024,1,64)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    #(32,1024,1,64),论文第3格

    #T-NET(2),feature_transform_net
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
        #(32,64,64)
    end_points['transform'] = transform

    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    #tf.squeeze()删掉为1的维度,此处[2]轴维度为1
    #(32,1024,64)*(64,64)->(32,1024,64)论文第4格

    net_transformed = tf.expand_dims(net_transformed, [2])
    #(32,1024,1,64)

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    #(32,1024,1,1024)论文第5格

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    #stride=[2, 2]?
    #(32,1024,1,1024)->(32,1,1,1024),论文第六格
    net = tf.reshape(net, [batch_size, -1])
    # (32,1024)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    #(32,40),论文第7格
    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B,
        A regularization loss (withweight 0.001)
        is added to the softmax classification loss to
        make the matrix close to orthogonal."""
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    #L=||I-AAt||f2
    transform = end_points['transform'] # BxKxK,(32,64,64)
    K = transform.get_shape()[1].value
    #tf.transpose()将a进行转置，并且根据perm参数重新排列输出维度。
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    #(32,64,64)*(32,64,64)  (AAt)
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    #Lreg=
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
