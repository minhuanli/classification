# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:30:56 2018

@author: HP
"""

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import os
from skimage import io

def boxpixel1(data,xrange,yrange,zrange,background,rdshift=0):
    xc=np.int32(np.round(data[0]))
    yc=np.int32(np.round(data[1]))
    zc=np.int32(np.round(data[2]))
    
    if rdshift != 0:
        shiftx = np.random.randint(-rdshift,high=rdshift)
        shifty = np.random.randint(-rdshift,high=rdshift)
        xc = xc + shiftx
        yc = yc + shifty
    
    
    if(xc-xrange > 0 and yc-yrange > 0 and zc-zrange > 0 and xc+xrange < 512 and yc+yrange < 512 and zc+zrange < 512):
        temp=background[(zc-zrange):(zc+zrange+1),(yc-yrange):(yc+yrange+1),(xc-xrange):(xc+xrange+1)]
        
    temp = np.reshape(temp,[-1])
    return temp

def boxpixelall(data,xrange,yrange,zrange,background,rdshift=0):
    nn = data.shape[0]
    psize = (2*xrange + 1)*(2*yrange+1)*(2*zrange+1)
    result = np.zeros([nn,psize])
    for i in range(nn):
        result[i,:] = boxpixel1(data[i,:],xrange,yrange,zrange,background,rdshift=0)
        
    return result

'''
pic0 = np.int32(io.imread(r'D:\liminhuan\machine learning\20180424\datafortrain\pic0_512.tiff'))
pic29 = np.int32(io.imread(r'D:\liminhuan\machine learning\20180424\datafortrain\pic29_512.tiff'))

b0c = np.loadtxt(r'D:\liminhuan\machine learning\20180424\datafortrain\b0c_512.txt',dtype =np.float32)
b29c = np.loadtxt(r'D:\liminhuan\machine learning\20180424\datafortrain\b29c_512.txt',dtype =np.float32)

w0 = np.where(b29c[:,5] > 0.40)
sld = b29c[w0]
w1 = np.where(b0c[:,5] < 0.15)
liq = b0c[w1]
index0 = np.random.choice(sld.shape[0],size=7000,replace=False)
index1 = np.random.choice(liq.shape[0],size=7000,replace=False)
slddata1st = boxpixelall(sld[index0,:],14,14,9,pic29)
liqdata1st = boxpixelall(liq[index1,:],14,14,9,pic0)

'''
