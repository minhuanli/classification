# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:47:03 2018

@author: Minhuan Li  minhuanli@g.harvard.edu
"""

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
from skimage import io

# if u want to retrain the network parameters, just denote the first 3 part, and activate the checkpoint load line

classnum = 2
batchsize = 64
#feature = 87
#droprate = 0.3
droprate = 0.3
model_dir = r'D:\liminhuan\machine learning\20180424\ckpointastous_512'
####################################
## to deal with the labels and calculate the entropy

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

####################################
'''
def image_cut(posinfo,originimage,trainnum = 4000, testnum = 2000):
 ## initialization
    w = np.where(posinfo[:,5] > 0.35)
    sldpos = posinfo[w,:]
    w = np.where(posinfo[:,5] < 0.15)
    liqpos = posinfo[w,:]
    sldpic = np.float32[-1,27*27*11]
    liqpic = np.float32[-1,27*27*11]
    sldsize = np.size(sldpos,axis = 0) 
    liqsize = np.size(liqpos,axis = 0)

## cut the picture from the original pic and reshape them
    for i in range(sldsize):
        xstart = sldpos[i,0] - 13
        xend = sldpos[i,0] + 13
        ystart = sldpos[i,1] - 13
        yend = sldpos[i,1] + 13
        zstart = sldpos[i,2] - 5
        zend = sldpos[i,2] + 5
        sldpic[i,:] = np.reshape(originimage[zstart:zend,xstart:xend,ystart:yend],[1,-1])
        
    
    for i in range(liqsize):
        xstart = liqpos[i,0] - 13
        xend = liqpos[i,0] + 13
        ystart = liqpos[i,1] - 13
        yend = liqpos[i,1] + 13
        zstart = liqpos[i,2] - 5
        zend = liqpos[i,2] + 5
        liqpic[i,:] = np.reshape(originimage[zstart:zend,xstart:xend,ystart:yend],[1,-1])        
        
## make training data according to the given number
    trainsldselect = np.random.choice(sldpic.shape[0],size = trainnum)    
    trainliqselect = np.random.choice(liqpic.shape[0],size = trainnum)
    train_data = np.float32(sldpic[trainsldselect,:] + liqpic[trainliqselect,:])
    
## make training label according to the givin number
    train_label = np.float32(np.ones[trainnum] + np.zeros[trainnum])
    
## make eval data according to the givin number
    evalsldselect = np.random.choice(sldpic.shape[0],size = testnum)    
    evalliqselect = np.random.choice(liqpic.shape[0],size = testnum)
    eval_data = np.float32(sldpic[evalsldselect,:] + liqpic[evalliqselect,:])
    
## make eval label according to the givin number
    eval_label = np.float32(np.ones[testnum] + np.zeros[testnum])
    
    return train_data,train_label,eval_data,eval_label
'''
####################################
def predict_batch(evaldata,batchsize=1000):
    nn = np.size(evaldata,axis=0)
    batchnum = np.int(np.ceil(nn/batchsize))
    result = np.ones(nn)

    for i in range(batchnum):
        start = i*batchsize
        end = min((i+1)*batchsize,nn)
        evaldatat = evaldata[start:end,:]
        evallogits = sess.run(logits,{xs:evaldatat})
        #softmax = cal_softmax(evallogits)
        prediction = np.argmax(evallogits,axis=1)
        result[start:end]=prediction
        
    return result
####################################
def loss_batch(evaldata,evallabel,batchsize=1000):
    nn = np.size(evaldata,axis=0)
    batchnum = np.int(np.ceil(nn/batchsize))
    result = np.ones(batchnum)

    for i in range(batchnum):
        start = i*batchsize
        end = min((i+1)*batchsize,nn)
        evaldatat = evaldata[start:end,:]
        evallabelt = evallabel[start:end]
        evallosst = sess.run(loss,{xs:evaldatat,ys:evallabelt})
        #softmax = cal_softmax(evallogits)
        result[i]= evallosst
        
    return np.mean(result)

#######################  I . load data part    ####################################
## Liq data first
## 0 for liq and 1 for sld
#train_data = trainData
train_data =  np.concatenate( (slddata1st[0:5000,:],liqdata1st[0:5000,:]) ) 
  
#train_labels = trainLabel
#train_labels = np.asarray(trainLabel, dtype=np.int32)
train_labels = np.asarray(np.concatenate((np.zeros(5000),np.ones(5000))),dtype=np.int32)
#eval_data = evalData
eval_data =  np.concatenate( (slddata1st[5000:7000,:],liqdata1st[5000:7000,:]) )
  
#eval_labels = evalLabel
#eval_labels = np.asarray(evalLabel, dtype=np.int32)
eval_labels = np.asarray(np.concatenate((np.zeros(2000),np.ones(2000))),dtype=np.int32)


#train_data = plt.imread(r'D:\liminhuan\machine learning\bpass_data\bpass_train.tiff')  # Returns np.array
#train_data = np.asarray(train_data, dtype=np.float32)
  
#train_labels = np.loadtxt(r'D:\liminhuan\machine learning\bpass_data\bpass label.txt')
#train_labels = np.asarray(train_labels, dtype=np.int32)
  
#eval_data = plt.imread(r'D:\liminhuan\machine learning\bpass_data\pass_eval.tiff')  # Returns np.array
#eval_data = np.asarray(eval_data, dtype=np.float32)
  
#eval_labels = np.loadtxt(r'D:\liminhuan\machine learning\bpass_data\eval label.txt')
#eval_labels = np.asarray(eval_labels, dtype=np.int32)

print(1)
memory_capacity = np.shape(train_data)[0]
print(2)
## set the input train data and train label 
#xs = tf.placeholder(tf.float32,[None,53*53*19])
xs = tf.placeholder(tf.float32,[None,29*29*19])
ys = tf.placeholder(tf.int32,[None])


###################################### II . network structure part ############################
# input is the origin feature
# the common output is logits layer, without softmax process
# if mode is set to be 2, the output is prediction class    
#input_layer = tf.reshape(xs, [-1, 19, 53, 53, 1])
input_layer = tf.reshape(xs,[-1,19,29,29,1])

#########################################################
# Convolutional Layer #0
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the pariticle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 41, 41, 41, 1]
# Output Tensor Shape: [batch_size, 41,41,41,1]

conv01 = tf.layers.conv3d(
    inputs=input_layer,
    filters=16,
    ## The original shape is [6,8,8]
    kernel_size=[5,7,7],
    strides=[1,1,1],
    padding="same",
    dilation_rate=[1,1,1],
    activation=tf.nn.relu,name = 'aslay1')

# Convolutional Layer #0
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the pariticle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 41, 41, 41, 1]
# Output Tensor Shape: [batch_size, 41,41,41,1]
conv02 = tf.layers.conv3d(
    inputs=conv01,
    filters=16,
    ## The original shape is [6,8,8]
    kernel_size=[5,7,7],
    strides=[1,1,1],
    padding="same",
    dilation_rate=[2,2,2],
    activation=tf.nn.relu,name = 'aslay2')

# Convolutional Layer #0
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the pariticle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 41, 41, 41, 1]
# Output Tensor Shape: [batch_size, 41,41,41,1]
conv03 = tf.layers.conv3d(
    inputs=conv02,
    filters=16,
    ## The original shape is [6,8,8]
    kernel_size=[5,7,7],
    strides=[1,1,1],
    padding="same",
    dilation_rate=[4,4,4],
    activation=tf.nn.relu,name = 'aslay3')

# Convolutional Layer #0
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the pariticle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 41, 41, 41, 1]
# Output Tensor Shape: [batch_size, 41,41,41,1]

#conv04 = tf.layers.conv3d(
#    inputs=conv03,
#    filters=16,
#    ## The original shape is [6,8,8]
#    kernel_size=[3,10,10],
#    strides=[1,1,1],
#    padding="same",
#    dilation_rate=[8,8,8],
#    activation=tf.nn.relu,name = 'aslay4')

##########################################################

# Convolutional Layer #1
# Computes 32 features using a 6*8*8 filter with ReLU activation.
# kernel size correspond to the pariticle size 
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 41, 41, 41, 1]
# Output Tensor Shape: [batch_size, 18, 17, 17, 32]
conv1 = tf.layers.conv3d(
    inputs=conv03,
    filters=32,
    ## The original shape is [6,8,8]
    kernel_size=[5,7,7],
    strides=[2,3,3],
    padding="valid",
    activation=tf.nn.relu,name = 'c3layer1')
# Pooling Layer #1
# First max pooling layer with a 3*2*2 filter and stride of 2
# Input Tensor Shape: [batch_size, 18, 17, 17, 32]
# Output Tensor Shape: [batch_size, 8, 8, 8, 32]
pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=1, padding="same",name='player1')
# Convolutional Layer #2
# Computes 64 features using a 3*3*3 filter.
# kernel size here correspond to the first neighbor shell
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 8, 8, 8, 32]
# Output Tensor Shape: [batch_size, 8, 8, 8, 64]
conv2 = tf.layers.conv3d(
    inputs=pool1,
    filters=64,
    kernel_size=[2,2,2],
    padding="valid",
    activation=tf.nn.relu,name='c3layer2')

# Pooling Layer #2
# Second max pooling layer with a 2*3*3 filter and stride of 2
# Input Tensor Shape: [batch_size, 8, 8, 8, 64]
# Output Tensor Shape: [batch_size, 4, 4, 4, 64]
pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2, padding="same",name='player2')

# Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 7, 7, 64]
# Output Tensor Shape: [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 4 * 4 * 4 * 64])

# Dense Layer
# Densely connected layer with 1024 neurons
# Input Tensor Shape: [batch_size, ]
# Output Tensor Shape: [batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu,name='dlayer1')

# Add dropout operation; 0.6 probability that element will be kept
dropout = tf.layers.dropout(
        inputs=dense, rate=droprate)

# Logits layer
# Input Tensor Shape: [batch_size, 1024]
# Output Tensor Shape: [batch_size, classnum]
logits = tf.layers.dense(inputs=dropout, units=classnum,name='logitlayer1')
  

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
model_path = os.path.join(model_dir,'modeltest.ckpt')

#saver.restore(sess,model_path)   # load the trained checkpoint file for further train

######################### V. train part ################################## 
for i in range(2001):
    indices = np.random.choice(memory_capacity,size=batchsize)
    sess.run(train,{xs:train_data[indices,:],ys:train_labels[indices]})
    if i % 200 == 0:
        losst = sess.run(loss,{xs:train_data[indices,:],ys:train_labels[indices]})
        print("steps: %s  loss: %s"%(i,losst))
        
saver.save(sess,model_path)

######################### VI. evaluate part ########################################################## 

#evallogits = sess.run(logits,{xs:eval_data})
#softmax = cal_softmax(evallogits)
prediction = predict_batch(evaldata=eval_data,batchsize=100)
eval_result = accuracy_cal(output = prediction, label = eval_labels)
#lossf = sess.run(loss,{xs:eval_data,ys:eval_labels})
lossf = loss_batch(evaldata=eval_data,evallabel=eval_labels,batchsize=100)
#evalresult = ses.run(eval_result,{ks: eval_labels})
print('final loss:%s'%lossf)
print('evaluation accuracy: %s'%eval_result)











