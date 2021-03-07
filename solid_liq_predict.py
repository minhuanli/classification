# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 13:20:32 2018

@author: HP
"""
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
from skimage import io

def sldliq_pred(pic,bc,model_dir,batchsize=3000):
    
# Returns np.array
    model_path = os.path.join(model_dir,'modeltest.ckpt')
    saver.restore(sess,model_path)

    nn = np.size(bc,axis=0)
    batchnum = np.int(np.ceil(nn/batchsize))
    result = np.ones(nn)

    for i in range(batchnum):
        start = i*batchsize
        end = min((i+1)*batchsize,nn)
        bct = bc[start:end,:]
        
        datat = boxpixelall(bct,26,26,9,pic,rdshift=0)
    
        evallogits = sess.run(logits,{xs:datat})
        #softmax = cal_softmax(evallogits)
        prediction = np.argmax(evallogits,axis=1)
        result[start:end]=prediction

    return result
#np.savetxt(r'D:\liminhuan\machine learning\20180424\test.txt',prediction,fmt='%2f \r\n')
