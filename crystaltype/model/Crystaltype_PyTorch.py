# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 2021

@author: Minhuan Li  minhuanli@g.harvard.edu
"""

import torch
import time, random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from PIL import Image


class CrystalNet(nn.Module): 
    def __init__(self, droprate=0.4, classnum=3):
        super(CrystalNet, self).__init__()
        self.conv = nn.Sequential(
            # 3D Conv layer #1, Kernel (6,8,8), Stride 2, Neuron Number 32
            # Input Shape [batch_size,1,31,31,31]
            # Output Shape [batch_size,32,13,12,12] 
            nn.Conv3d(1,32,(6,8,8),stride=2),
            # A Relu Activation
            nn.ReLU(),
            
            # Pooling Layer #1, Kernel (3,2,2), Stride 2
            # Input Shape [batch_size,32,13,13,12]
            # Output Shape [batch_size,32,6,6,6] 
            nn.MaxPool3d((3,2,2),stride=2),
            
            # 3D Conv layer #2, Kernel (3,3,3), Stride 1, Neuron Number 64
            # Input Shape [batch_size,32,6,6,6]
            # Output Shape [batch_size,64,6,6,6]
            nn.Conv3d(32,64,(3,3,3),stride=1,padding=1),
            # A Relu Activation
            nn.ReLU(),
            
            # Pooling Layer #2, Kernel (2,2,2), Stride 2
            # Input Shape [batch_size,64,6,6,6]
            # Output Shape [batch_size,64,3,3,3] 
            nn.MaxPool3d((2,2,2),stride=2),
        )
        
        self.dense = nn.Sequential(
            # Densely connected layer #1 with 256 neurons
            # Input Tensor Shape: [batch_size, 3 * 3 * 3 * 64]
            # Output Tensor Shape: [batch_size, 256]
            nn.Linear(64*3*3*3,256),
            # A Relu Activation
            nn.ReLU(),
            
            # Dropout Layer for a better training results
            nn.Dropout(p=droprate),
            
            # Densely connected layer #2 as logits layer
            # Input Tensor Shape: [batch_size, 256]
            # Output Tensor Shape: [batch_size, classnum]
            nn.Linear(256,classnum),
        )
    
    def forward(self,inputs):
        '''
        inputs Shape [batch_size,1,31,31,31]
        feature Shape [batch_size,64,3,3,3]
        output Shape [batch_size,classnum]
        '''
        feature = self.conv(inputs)
        output = self.dense(feature.view(inputs.shape[0], -1))
        return output
    
    def predict(self,inputs):
        '''
        Return the type predicted by the CrystalNet Model
        Shape: [batch_size]
        '''
        output = self.forward(inputs)
        return output.argmax(dim=1)


def data_iter(batch_size, features, labels):
    '''
    To make life easier, create a generator to load batched data
    '''
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # last batch might not be full
        yield features.index_select(0, j), labels.index_select(0, j)


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = nn.CrossEntropyLoss() # Use CrossEntroyLoss for this multi-class classification task
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter():
            X = X.to(device)
            y = y.to(device)
            y_hat = net.forward(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter():
            assert isinstance(net, torch.nn.Module)
            net.eval() # evaluation mode, turn off dropout
            acc_sum += (net.predict(X.to(device)) == y.to(device)).float().sum().cpu().item()
            net.train() # change back to training mode
            n += y.shape[0]
    return acc_sum / n



if __name__ == '__main__':
    print("PyTorch Version:",torch.__version__)
    print("Cuda Version:", torch.version.cuda)
    print("Is there a cuda-device(GPU)?",torch.cuda.is_available())
    Image.MAX_IMAGE_PIXELS = None # Now we can open large tiff images

    # Read in training dataset and test dataset
    train_data = torch.from_numpy(np.asarray(plt.imread(r'../data/crystaltypelocal_train.tiff'), dtype=np.float32))
    train_labels = torch.from_numpy(np.loadtxt(r'../data/crystaltype_label.txt')).type(torch.LongTensor)
    test_data = torch.from_numpy(np.asarray(plt.imread(r'../data/crystaltypelocal_eval.tiff'), dtype=np.float32))
    test_labels = torch.from_numpy(np.loadtxt(r'../data/crystaltype_evallabel.txt')).type(torch.LongTensor)

    # Define the dataset generator
    def train_iter():
        return data_iter(batchsize, train_data.view(train_data.shape[0],1,31,31,31), train_labels)
    def test_iter():
        return data_iter(batchsize, test_data.view(test_data.shape[0],1,31,31,31), test_labels)

    # Define some constants
    classnum = 3 # bcc, hcp, fcc, intotal 3 types
    batchsize = 128
    droprate = 0.4
    lr = 0.001
    num_epochs = 50

    # Initialize a CrystalNet Model
    crystalnet_model = CrystalNet(droprate=droprate,classnum=classnum)
    # Use Adam as the optimizer
    optimizer = torch.optim.Adam(crystalnet_model.parameters(),lr=lr)


    # Save the state_dict of the model
    torch.save(crystalnet_model.state_dict(), '../model/crystalmodel.pt')

    # Reload the model for future use
    # Reinitialize a model cass instance, and then load the dict saved above
    #crystalnet_model_reload = CrystalNet()
    #crystalnet_model_reload.load_state_dict(torch.load(r'../model/crystalmodel.pt'))



























