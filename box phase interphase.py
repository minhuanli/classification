# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:27:21 2018

@author: lmh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:47:11 2017

@author: Pop
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


#######################  I . load data part  ####################################

dataa = plt.imread(r'D:\liminhuan\machine learning\20180119\crbox.tif')
dataa = np.asarray(dataa, dtype=np.float32)
labela = np.loadtxt(r'D:\liminhuan\machine learning\20180119\crboxlabel.txt')
labela = np.asarray(labela, dtype=np.int32)

datab = plt.imread(r'D:\liminhuan\machine learning\20180119\liqbox_rs_0513p1029.tif')
datab = np.asarray(datab, dtype=np.float32)
labelb = np.loadtxt(r'D:\liminhuan\machine learning\20180119\liqlabel_rs.txt')
labelb = np.asarray(labelb, dtype=np.int32)

datac = plt.imread(r'D:\liminhuan\machine learning\20180119\shiftgbbox.tif')
datac = np.asarray(datac, dtype=np.float32)
labelc = np.loadtxt(r'D:\liminhuan\machine learning\20180119\gbboxlabel.txt')
labelc = np.asarray(labelc, dtype=np.int32)

indiceat = np.random.choice(np.shape(dataa)[0],size = 2500 , replace = False )
indiceae = np.asarray(list( set(np.arange(np.shape(dataa)[0]))  -set(indiceat) ), dtype = np.int32)

indicebt = np.random.choice(np.shape(datab)[0],size = 3000 , replace = False )
indicebe = np.asarray(list( set(np.arange(np.shape(datab)[0])) -set(indicebt) ), dtype = np.int32)

indicect = np.random.choice(np.shape(datac)[0],size = 2800 , replace = False )
indicece = np.asarray(list( set(np.arange(np.shape(datac)[0])) -set(indicect) ), dtype = np.int32)


train_data = np.vstack([dataa[indiceat,:],datab[indicebt,:],datac[indicect,:]])
train_labels = np.concatenate((labela[indiceat],labelb[indicebt],labelc[indicect]))


eval_data = np.vstack([dataa[indiceae,:],datab[indicebe,:],datac[indicece,:]])
eval_labels = np.concatenate((labela[indiceae],labelb[indicebe],labelc[indicece]))




print(1)
memory_capacity = np.shape(train_data)[0]
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
model_path = os.path.join(model_dir,'1modeltest1.ckpt')

saver.restore(sess,model_path)   # load the trained checkpoint file for further train

######################### V. train part ################################## 
'''
memory_capacity = np.shape(train_data)[0]
time_start = time.time()
for i in range(6001):
    indices = np.random.choice(memory_capacity,size=batchsize)
    sess.run(train,{xs:train_data[indices,:],ys:train_labels[indices]})
    if i % 200 == 0:
        time_now = time.time()
        losst = sess.run(loss,{xs:train_data[indices,:],ys:train_labels[indices]})
        print("steps: %s  loss: %s  time: %s s"%(i,losst,round(time_now-time_start,2)))
        
saver.save(sess,model_path)
'''
######################### VI. evaluate part ########################################################## 
indice0 = np.random.choice(np.size(eval_data[:,0]),size=1500, replace=False)
evallogits = sess.run(logits,{xs:eval_data[indice0,:]})
#softmax = cal_softmax(evallogits)
prediction = np.argmax(evallogits,axis=1)
eval_result = accuracy_cal(output = prediction, label = eval_labels[indice0])
lossf = sess.run(loss,{xs:eval_data[indice0,:],ys:eval_labels[indice0]})
#evalresult = ses.run(eval_result,{ks: eval_labels})
print('final loss:%s'%lossf)
print('evaluation accuracy: %s'%eval_result)

