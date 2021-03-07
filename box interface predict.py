# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:39:23 2018
predict interface with trained network
@author: HP
"""


import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
import time


# if u want to retrain the network parameters, just denote the first 3 part, and activate the checkpoint load line

classnum = 3
batchsize = 128
#feature = 87
droprate = 0.3
model_dir = r'D:\liminhuan\machine learning\20180119\NN1checkpoint'
####################################

def loss_cal(labels,output):
	 
   onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=classnum)
   onehot_labels = tf.reshape(onehot_labels,[-1,classnum])
   loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=output)
   
   return loss 

###################################
## calculate the prediction accuracy after the train part 
def accuracy_cal(output,label):
   temp = (output == label)
   true = sum(temp)
   allnum = np.shape(label)[0]
   accuracy = true / allnum
   
   return accuracy

####################################
def cal_softmax(logits):
    return np.transpose(np.exp(logits)) / np.sum(np.exp(logits), axis=1)

##############################################################################
#######################  I . load data part  ####################################


eval_data = plt.imread(r'D:\liminhuan\machine learning\20180119\lboxtest2.tif')
eval_data = np.asarray(eval_data, dtype=np.float32)
'''
print(2)
## set the input train data and train label 
xs = tf.placeholder(tf.float32,[None,41*61*61])
ys = tf.placeholder(tf.int32,[None])


###################### II . network structure part ############################
# input is the origin feature
# the common output is logits layer, without softmax process

input_layer = tf.reshape(xs, [-1, 41, 61, 61, 1])

# Convolutional Layer #1
# Computes 32 features using a 4*6*6 filter with ReLU activation.
# kernel size correspond to the particle size 
# Padding is added to preserve width and height. here i choose a valid padding
# Input Tensor Shape: [batch_size, 41, 61, 61, 1]
# Output Tensor Shape: [batch_size, 19, 19, 19, 32]
conv1 = tf.layers.conv3d(
    inputs=input_layer,
    filters=32,
    kernel_size=[4,6,6],
    strides=[2,3,3],
    padding="valid",
    activation=tf.nn.relu,name ='7c3layer3')
# Pooling Layer #1
# First max pooling layer with a 2*2*2 filter and stride of 2 
# Input Tensor Shape: [batch_size, 19, 19, 19, 32]
# Output Tensor Shape: [batch_size, 9, 9, 9, 32]
pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=2, padding="valid",name='7player4')

# Convolutional Layer #2
# Computes 64 features using a 3*3*3 filter.
# kernel size here correspond to the first neighbor shell
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 9, 9, 9, 32]
# Output Tensor Shape: [batch_size, 9, 9, 9, 64]
conv2 = tf.layers.conv3d(
    inputs=pool1,
    filters=64,
    kernel_size=[9,9,9],
    padding="same",
    activation=tf.nn.relu,name='7c3layer5')

# Pooling Layer #2
# Second max pooling layer with a 2*2*2 filter and stride of 2
# Input Tensor Shape: [batch_size, 9, 9, 9, 64]
# Output Tensor Shape: [batch_size, 5, 5, 5, 64]
pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2, padding="same",name='7player6')


# Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 5, 5, 5, 64]
# Output Tensor Shape: [batch_size, 5 * 5 * 5 * 64]
pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 5 * 64])

# Dense Layer
# Densely connected layer with 1024 neurons
# Input Tensor Shape: [batch_size, 5 * 5 * 5 * 64]
# Output Tensor Shape: [batch_size, 256]
dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu,name='7dlayer7')

# Add dropout operation; 1-droprate probability that element will be kept
dropout = tf.layers.dropout(
    inputs=dense, rate=droprate)

# Logits layer
# Input Tensor Shape: [batch_size, 256]
# Output Tensor Shape: [batch_size, classnum]
logits = tf.layers.dense(inputs=dropout, units=classnum,name='7logitlayer8')

############################## III. loss and train claim #########################################
## calculate the loss between logits output and label, in train mode 
loss = loss_cal(labels = ys,output = logits)
print(4)
## train the network parameters with loss BP
optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(loss)
print(5)

#####################IV. initialization part ###########################
sess = tf.Session() 
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
print(6)
'''
model_path = os.path.join(model_dir,'1modeltest1.ckpt')
saver.restore(sess,model_path)   # load the trained checkpoint file for further train
evallogits = sess.run(logits,{xs:eval_data})
prediction = np.argmax(evallogits,axis=1)





