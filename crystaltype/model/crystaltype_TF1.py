# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:47:11 2017

@author: Minhuan Li  minhuanli@g.harvard.edu
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
droprate = 0.4
#model_dir = r'D:\liminhuan\machine learning'
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

#######################  I . load data part  ####################################

train_data = plt.imread(r'D:\liminhuan\machine learning\crystaltypelocal_train.tiff')  # Returns np.array
train_data = np.asarray(train_data, dtype=np.float32)
  
train_labels = np.loadtxt(r'D:\liminhuan\machine learning\crystaltype_label.txt')
train_labels = np.asarray(train_labels, dtype=np.int32)
  
eval_data = plt.imread(r'D:\liminhuan\machine learning\crystaltypelocal_eval.tiff')  # Returns np.array
eval_data = np.asarray(eval_data, dtype=np.float32)
  
eval_labels = np.loadtxt(r'D:\liminhuan\machine learning\crystaltype_evallabel.txt')
eval_labels = np.asarray(eval_labels, dtype=np.int32)
  
print(1)
memory_capacity = np.shape(train_data)[0]
print(2)
## set the input train data and train label 
xs = tf.placeholder(tf.float32,[None,31*31*31])
ys = tf.placeholder(tf.int32,[None])


###################### II . network structure part ############################
# input is the origin feature
# the common output is logits layer, without softmax process

input_layer = tf.reshape(xs, [-1, 31, 31, 31, 1])

# Convolutional Layer #1
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the particle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 31, 31, 31, 1]
# Output Tensor Shape: [batch_size, 13, 12, 12, 32]
conv1 = tf.layers.conv3d(
    inputs=input_layer,
    filters=32,
    kernel_size=[6,8,8],
    strides=[2,2,2],
    padding="valid",
    activation=tf.nn.relu,name ='7c3layer312')
# Pooling Layer #1
# First max pooling layer with a 3*2*2 filter and stride of 2 
# Input Tensor Shape: [batch_size, 13, 12, 12, 32]
# Output Tensor Shape: [batch_size, 6, 6, 6, 32]
pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[3, 2, 2], strides=2, padding="valid",name='7player412')

# Convolutional Layer #2
# Computes 64 features using a 3*3*3 filter.
# kernel size here correspond to the first neighbor shell
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 6, 6, 6, 32]
# Output Tensor Shape: [batch_size, 6, 6, 6, 64]
conv2 = tf.layers.conv3d(
    inputs=pool1,
    filters=64,
    kernel_size=[2,2,2],
    padding="same",
    activation=tf.nn.relu,name='7c3layer512')

# Pooling Layer #2
# Second max pooling layer with a 2*2*2 filter and stride of 2
# Input Tensor Shape: [batch_size, 6, 6, 6, 64]
# Output Tensor Shape: [batch_size, 3, 3, 3, 64]
pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2, padding="same",name='7player612')


# Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 3, 3, 3, 64]
# Output Tensor Shape: [batch_size, 3 * 3 * 3 * 64]
pool2_flat = tf.reshape(pool2, [-1, 3 * 3 * 3 * 64])

# Dense Layer
# Densely connected layer with 1024 neurons
# Input Tensor Shape: [batch_size, 3 * 3 * 3 * 64]
# Output Tensor Shape: [batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu,name='7dlayer712')

# Add dropout operation; 1-droprate probability that element will be kept
dropout = tf.layers.dropout(
    inputs=dense, rate=droprate)

# Logits layer
# Input Tensor Shape: [batch_size, 1024]
# Output Tensor Shape: [batch_size, classnum]
logits = tf.layers.dense(inputs=dropout, units=classnum,name='7logitlayer812')

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
#model_path = os.path.join(model_dir,'1modeltest.ckpt')

#saver.restore(sess,model_path)   # load the trained checkpoint file for further train

######################### V. train part ################################## 
memory_capacity = np.shape(train_data)[0]
time_start = time.time()
for i in range(5001):
    indices = np.random.choice(memory_capacity,size=batchsize)
    sess.run(train,{xs:train_data[indices,:],ys:train_labels[indices]})
    if i % 100 == 0:
        time_now = time.time()
        losst = sess.run(loss,{xs:train_data[indices,:],ys:train_labels[indices]})
        print("steps: %s  loss: %s  time: %s s"%(i,losst,round(time_now-time_start,2)))
        
#saver.save(sess,model_path)
######################### VI. evaluate part ########################################################## 
evallogits = sess.run(logits,{xs:eval_data})
#softmax = cal_softmax(evallogits)
prediction = np.argmax(evallogits,axis=1)
eval_result = accuracy_cal(output = prediction, label = eval_labels)
lossf = sess.run(loss,{xs:eval_data,ys:eval_labels})
#evalresult = ses.run(eval_result,{ks: eval_labels})
print('final loss:%s'%lossf)
print('evaluation accuracy: %s'%eval_result)

