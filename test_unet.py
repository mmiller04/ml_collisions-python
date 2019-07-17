#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:01:50 2019

@author: AndresMillerHernandez
"""

### data preprocessing

from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit
import math
from torch.utils.data import Dataset, DataLoader
from scipy import stats

### hyperparams

batch_size = 10
lr = 0.0005
momentum = 0.99
num_epochs = 1
percentage_train = 0.85
lr_decay = 1
step_size = 30
loss_weights = [1,0,0,0]

### neural network setup

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=True)

class UnetDownBlock(nn.Module):
   
    def __init__(self, inplanes, planes, predownsample_block):
        
        super(UnetDownBlock, self).__init__()
        
        self.predownsample_block = predownsample_block
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
    def forward(self, x):
        
        x = self.predownsample_block(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x
    
class UnetUpBlock(nn.Module):
   
    def __init__(self, inplanes, planes, postupsample_block=None):
        
        super(UnetUpBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        if postupsample_block is None: 
            
            self.postupsample_block = nn.ConvTranspose2d(in_channels=planes,
                                                               out_channels=planes//2,
                                                               kernel_size=2,
                                                               stride=2)
            
        else:
            
            self.postupsample_block = postupsample_block
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.postupsample_block(x)
        
        return x
    
    
class Unet(nn.Module):
    
    def __init__(self):
        
        super(Unet, self).__init__()
        
        self.predownsample_block = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.identity_block = nn.Sequential()
        
        self.block1 = UnetDownBlock(
                                    predownsample_block=self.identity_block,
                                    inplanes=2, planes=64
                                    )
        
        self.block2_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=64, planes=128
                                         )
        
        self.block3_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=128, planes=256
                                         )

        self.block4_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=256, planes=512
                                         )
        
        self.block5_down = UnetDownBlock(
                                         predownsample_block=self.predownsample_block,
                                         inplanes=512, planes=1024
                                         )
        
        self.block1_up = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                                  kernel_size=2, stride=2)
        
        self.block2_up = UnetUpBlock(
                                     inplanes=1024, planes=512
                                     )
        
        self.block3_up = UnetUpBlock(
                                     inplanes=512, planes=256
                                     )
        
        self.block4_up = UnetUpBlock(
                                     inplanes=256, planes=128
                                     )
        
        self.block5 = UnetUpBlock(
                                  inplanes=128, planes=64,
                                  postupsample_block=self.identity_block
                                  )
        
        self.logit_conv = nn.Conv2d(
                                    in_channels=64, out_channels=1, kernel_size=1,
                                    )
        
        
    def forward(self, x):
        
        features_1s_down = self.block1(x)
        features_2s_down = self.block2_down(features_1s_down)
        features_4s_down = self.block3_down(features_2s_down)
        features_8s_down = self.block4_down(features_4s_down)
        
        features_16s = self.block5_down(features_8s_down)
        
        features_8s_up = self.block1_up(features_16s)
        features_8s_up = torch.cat([features_8s_down, features_8s_up],dim=1)
        
        features_4s_up = self.block2_up(features_8s_up)
        features_4s_up = torch.cat([features_4s_down, features_4s_up],dim=1)
        
        features_2s_up = self.block3_up(features_4s_up)
        features_2s_up = torch.cat([features_2s_down, features_2s_up],dim=1)
        
        features_1s_up = self.block4_up(features_2s_up)
        features_1s_up = torch.cat([features_1s_down, features_1s_up],dim=1)
        
        features_final = self.block5(features_1s_up)
        
        logits = self.logit_conv(features_final)
        
        return logits

net = Unet()

### Read from .txt file

def load_data():
    fid1 = open('medium_test/large_fi_test_network2.txt','r')
    fid2 = open('medium_test/large_fe_test_network2.txt','r')
    fid3 = open('medium_test/large_dfi_test_network2.txt','r')
    fid4 = open('medium_test/large_dfe_test_network2.txt','r')

    fid_vol_i = open('voli.txt','r')
    fid_vol_e = open('vole.txt','r')
    
    big_list_fi = fid1.readlines()
    big_list_fe = fid2.readlines()
    big_list_dfi = fid3.readlines()
    big_list_dfe = fid4.readlines()
    num_nodes = int(len(big_list_fi)/32)
    
    f = np.empty([int(num_nodes),2,32,32])
    df = np.empty([int(num_nodes),2,32,32])
    
    for i in range(num_nodes):
        for j in range(32):
            each_vpar_fi = big_list_fi[i*32+j].split(',')
            each_vpar_fe = big_list_fe[i*32+j].split(',')
            each_vpar_dfi = big_list_dfi[i*32+j].split(',')
            each_vpar_dfe = big_list_dfe[i*32+j].split(',')

            f[i,0,j,0] = float(each_vpar_fi[0][1:])
            f[i,1,j,0] = float(each_vpar_fe[0][1:])
            df[i,0,j,0] = float(each_vpar_dfi[0][1:])
            df[i,0,j,1] = float(each_vpar_dfe[0][1:])

            for k in range(1,len(each_vpar_fi)-1,1):
                f[i,0,j,k] = float(each_vpar_fi[k])
                f[i,1,j,k] = float(each_vpar_fe[k])
                df[i,0,j,k] = float(each_vpar_dfi[k])
                df[i,1,j,k] = float(each_vpar_dfe[k])
                
            f[i,0,j,30] = float(each_vpar_fi[30][:-2])
            f[i,1,j,30] = float(each_vpar_fe[30][:-2])
            df[i,0,j,30] = float(each_vpar_dfi[30][:-2])
            df[i,1,j,30] = float(each_vpar_dfe[30][:-2])
            
            f[i,0,j,31] = float(each_vpar_fi[30][:-2])
            f[i,1,j,31] = float(each_vpar_fe[30][:-2])
            df[i,0,j,31] = float(each_vpar_dfi[30][:-2])
            df[i,1,j,31] = float(each_vpar_dfe[30][:-2])

    df+=f

    for z in range(num_nodes):
        f[z,0,:,:] = stats.zscore(f[z,0,:,:])
        f[z,1,:,:] = stats.zscore(f[z,1,:,:])
        df[z,0,:,:] = stats.zscore(df[z,0,:,:])
        df[z,1,:,:] = stats.zscore(df[z,1,:,:])

        
    vol_i = []
    vol_e = []
    for vol in fid_vol_i.readlines():
        vol_i.append(float(vol[:-1]))
    for vol in fid_vol_e.readlines():
        vol_e.append(float(vol[:-1]))  
        
    return f,df,num_nodes,vol_i,vol_e
        
class DistFuncDataset(Dataset):
    
    def __init__(self, f_array, df_array, transform=None):
        self.data = torch.from_numpy(f_array).float()
        self.target = torch.from_numpy(df_array).float()
        self.transform = transform
        
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        y = y.view(-1,32,32)
    
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    
def split_data(f,df,num_nodes):
    
    # shuffle data up
    expand_df = np.expand_dims(df,axis=1)
    all_data = np.concatenate((f,expand_df),axis=1)
    np.random.shuffle(all_data)
    
    f = all_data[:,:2,:,:]
    df = all_data[:,2,:,:]
    
    num_train=int(np.floor(percentage_train*num_nodes))
    num_test = num_nodes - num_train
    
    f_train = f[:num_train,:,:,:]
    f_test = f[num_train:,:,:,:]
    
    df_train = df[:num_train,:,:]
    df_test = df[:num_train,:,:]
    
    trainset=DistFuncDataset(f_train, df_train)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    
    testset=DistFuncDataset(f_test, df_test)
    
    testloader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)

    return trainloader, testloader, f_test, num_train, num_test

### check conservation properties before testing to calculate accuracy

def check_properties(f_slice, vol):
    mass = 0
    momentum = 0
    energy = 0
    
    npar = len(f_slice[0,:])-1
    nperp = len(f_slice[:,0])
    
    f0_smu_max = 4
    f0_vp_max = 4
    
    vpar = np.linspace(0,f0_smu_max,npar) # 2*f0_nvp + 1
    vperp = np.linspace(-f0_vp_max,f0_vp_max,nperp) # f0_nmu + 1
    
    for i in range(nperp):
        for j in range(npar):
            mass += f_slice[i,j]*vol[i]
            momentum += vpar[j]*f_slice[i,j]*vol[i]
            energy += (vpar[j]**2 + vperp[i]**2)*f_slice[i,j]*vol[i]
            
    return mass, momentum, energy
    
def conservation_before(f,vol,num_test,sp_flag):
    
    mass_in = []
    momentum_in = []
    energy_in = []

    for n in range(num_test):
        
        mass, momentum, energy = check_properties(abs(f[n,sp_flag,:,:]),vol)
                
        mass_in.append(mass)         
        momentum_in.append(momentum)         
        energy_in.append(energy)         
    
    cons_in = np.array([mass_in, momentum_in, energy_in])
    
    return cons_in


"""
for batch_idx, (data, target) in enumerate(trainloader):
    print('Batch idx {}, data shape {}, target shape {}'.format(
            batch_idx, data.shape, target.shape))


dataiter = iter(trainloader)
data, target = dataiter.next()

plt.contourf(initial[0,:,:,1])
plt.colorbar()
plt.show()
"""

### train network

def train(trainloader,net,vol):
    
    loss_vector=[]
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        for i, (data, targets) in enumerate(trainloader):
    
            optimizer.zero_grad()
            outputs = net(data)

            cons_loss = torch.zeros(3)

            for n in range(batch_size):
                mass_b,mom_b,energy_b = check_properties(abs(data[n,0,:,:]),vol)
                mass_a,mom_a,energy_a = check_properties(abs(outputs[n,0,:,:]),vol)
                
                cons_loss[0] += abs((mass_a - mass_b)/mass_b)
                cons_loss[1] += abs((mom_a - mom_b)/mom_b)
                cons_loss[2] += abs((energy_a - energy_b)/energy_b)

            l2_loss = criterion(outputs, targets)
            loss = criterion(outputs, targets)

            loss = l2_loss*loss_weights[0] \
            + cons_loss[0]*loss_weights[1] \
            + cons_loss[1]*loss_weights[2] \
            + cons_loss[2]*loss_weights[3] \
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))
                loss_vector.append(running_loss)
                running_loss = 0.0
                    
    iterations=np.linspace(1,len(loss_vector),len(loss_vector))
    
    plt.plot(iterations,loss_vector)
    
    return loss_vector, iterations 


### test the data (check conservation again)
   
def test(testloader,net,cons_in,vol,num_test):
    
    mass_out = []
    momentum_out = []
    energy_out = []
    
    l2_error=[]
    lt1=0
    gt1=0
    with torch.no_grad():
        for (data, targets) in testloader:
            outputs = net(data)
            
            for n in range(batch_size):
                mass, momentum, energy = check_properties(abs(outputs[n,0,:,:]),vol)
                
                mass_out.append(mass)         
                momentum_out.append(momentum)         
                energy_out.append(energy)         
            
            loss = criterion(outputs, targets)
            l2_error.append(loss.item()*100)
            if loss.item()*100 < 1:
                lt1+=1
            else:
                gt1+=1
              
    cons_out = np.array([mass_out, momentum_out, energy_out])

    cons_error = np.zeros([3,num_test])
    for c in range(num_test):
        cons_error[0,c] += (cons_out[0,c] - cons_in[0,c])/cons_in[0,c]
        cons_error[1,c] += (cons_out[1,c] - cons_in[1,c])/cons_in[1,c]
        cons_error[2,c] += (cons_out[2,c] - cons_in[2,c])/cons_in[2,c]
    
    print('Finished testing')
    print('Percentage with MSE<1: %d %%' % (
            100 * lt1/(lt1+gt1)))
    print('Percent error in conservation properties:\nmass: \
            %d %%\nmomentum: %d %%\nenergy: %d %%' % ( 
            100*cons_error[0,:].max(), 
            100*cons_error[1,:].max(), 
            100*cons_error[2,:].max()))
    
    return l2_error, cons_error


def test_perturb(f_slice, net):
    
    for i in range(14,17):
        for j in range(14,17):
            f_slice[:,i,j]=2*f_slice[:,i,j]
    
    f_to_return = f_slice[1,:,:]
    c
    for i in range(100):
        f_slice = net(f_slice)
        f_slice = np.concatenate(f_slice,f_slice)

    return f_to_return, f_slice


if __name__ == "__main__":
    start = timeit.default_timer()
    
    print('Loading data')
    f,df,num_nodes,vol_i,vol_e = load_data()
        
    print('Splitting data') 
    trainloader_i,testloader_i,f_testi,num_train,num_test = \
        split_data(f,df[:,0,:,:],num_nodes)
    trainloader_e,testloader_e,f_teste,num_train,num_test = \
        split_data(f,df[:,1,:,:],num_nodes)
    
    print('Calculating conservation properties')
    consi_in = conservation_before(f_testi,vol_i,num_test,0)
    conse_in = conservation_before(f_teste,vol_e,num_test,1)
        
    ### loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=lr_decay)
    
    print('Starting training')
    loss_vector_i,iterations_i = train(trainloader_i,net,vol_i)
    loss_vector_e,iterations_e = train(trainloader_e,net,vol_e)
    print('Finished training')
    
    print('Starting testing')
    l2_error_i, cons_error_i = test(testloader_i,net,consi_in,vol_i,num_test)
    l2_error_e, cons_error_e = test(testloader_e,net,conse_in,vol_e,num_test)

    stop = timeit.default_timer()
    
    print('Runtime: ' + str((stop-start)/3600) + 'hrs')
    """
    fid = open('lr_tune.txt','a')
    fid.write('lr: '+str(lr)+'\n')
    fid.write('runtime: '+str(stop-start)+'\n')
    for loss in loss_vector:
        fid.write(str(loss)+'\n')
    fid.close()
    """
    
    f_perturbed, f_corrected = test_perturb(f[0,:,:,:],net)

    plt.figure(1)
    plt.contourf(f_perturbed,100)
    
    plt.figure(2)
    plt.contourf(f_corrected,100)
    
