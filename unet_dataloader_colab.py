#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:46:52 2019

@author: AndresMillerHernandez
"""

from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import timeit
import h5py
import torch.utils.data as data
from scipy import stats
from google.colab import files

batch_size = 32
lr = 0.0005
momentum = 0.99
num_epochs = 10
percentage_train = 0.85
lr_decay = 0.01
step_size = 1
loss_weights = [1,1,1,1]
nphi = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

net_e = Unet().to(device)
net_i = Unet().to(device)


class DistFuncDataset(Dataset):
    
    def __init__(self, hf_f, hf_df, hf_vol:
        self.hf_f = hf_f
        self.hf_df = hf_df
        self.hf_vol = hf_vol
        
        nphi,nperp,ngrid,npar = hf_f.get('i_f').shape
        
        self.ngrid = ngrid
        self.nphi = nphi
        
    def __len__(self):
        return self.ngrid*self.nphi
     
    def __getitem__(self, index):
        iphi = index//self.ngrid         
        igrid = index - iphi*self.ngrid
                
        xi = self.hf_f['i_f'][iphi,:,igrid,:]
        xe = self.hf_f['e_f'][iphi,:,igrid,:]
        
        x = np.concatenate((np.expand_dims(xi,axis=0),np.expand_dims(xe,axis=0)),axis=0)     
        y = self.hf_df['i_df'][iphi,:,igrid,:]
        z = self.hf_vol['voli'][iphi,:,igrid]
            
        return x, y, z
    

def load_data():

  hf_f = h5py.File('/content/hdf5_data/hdf_f.h5','r')
  hf_df = h5py.File('/content/hdf5_data/hdf_df.h5','r')
  hf_vol = h5py.File('/content/hdf5_data/hdf_vol.h5','r')

  #### NORMALIZATION !!!
  
  dataset = DistFuncDataset(hf_f, hf_df, hf_vol)

  return dataset


def split_data(dataset, percentage_train):
  
  inds = np.arange(len(dataset))
  np.random.shuffle(inds) 
  
  num_train = int(np.floor(len(dataset)*percentage_train))
  
  train_inds = inds[:num_train]
  test_inds = inds[num_train:]
  
  trainset = data.Subset(dataset, train_inds)
  testset = data.Subset(dataset, test_inds)
  
  trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, pin_memory=True, num_workers=4)
  
  testloader = data.DataLoader(testset, batch_size=batch_size,
                               shuffle=True, pin_memory=True, num_workers=4)

  return trainloader, testloader


def check_properties(f_slice, vol, device):
    
    f_slice = f_slice.double()
       
    if len(f_slice.shape) == 2:
      nperp, npar = f_slice.shape
      nbatch = 1
    elif len(f_slice.shape) == 3:  
      nbatch,nperp,npar = f_slice.shape
          
    f0_smu_max = 4
    f0_vp_max = 4
    
    vpar = np.linspace(0,f0_smu_max,npar) # 2*f0_nvp + 1
    vperp = np.linspace(-f0_vp_max,f0_vp_max,nperp) # f0_nmu + 1
    
    vpar = torch.tensor(vpar).double().to(device)
    vperp = torch.tensor(vperp).double().to(device)
      
    short_ones = torch.ones(npar,nbatch,nperp).double().to(device)
    long_ones = torch.ones(nperp,nbatch,npar).double().to(device)
    
    vol_array = short_ones*vol.double()
    vol_array = vol_array.transpose_(0,1)
    vol_array = vol_array.transpose_(1,2)

    vpar_array = long_ones*vpar
    vpar_array = vpar_array.transpose_(0,1)
    
    vperp_array = short_ones*vperp
    vperp_array = vperp_array.transpose_(0,1)  
    vperp_array = vperp_array.transpose_(1,2)  
       
    mass_array = vol_array
    mom_array = vpar_array*vol_array
    energy_array = (vpar_array**2 + vperp_array**2)*vol_array
    
    mass_array, mom_array, energy_array = \
    mass_array.to(device), mom_array.to(device), energy_array.to(device)
    
    mass = (torch.sum(f_slice*mass_array).float())/nbatch
    momentum = (torch.sum(f_slice*mom_array).float())/nbatch
    energy = (torch.sum(f_slice*energy_array).float())/nbatch
            
    return mass, momentum, energy

def train(trainloader,net,device,sp_flag,loss_vector,epoch,iphi,end):
  
    props_before = []
    props_after = []
  
    for epoch in range(num_epochs):
      
      running_loss = 0.0
      for i, (data, targets, vol) in enumerate(trainloader):

          data, targets, vol = data.to(device), targets.to(device), vol.to(device)

          if sp_flag == 0:
              optimizer.zero_grad()
          else:
              optimizer_e.zero_grad()

          outputs = net(data)
          outputs = outputs.to(device)

          mass_b,mom_b,energy_b = check_properties(data[:,0,:,:-1],vol,device)
          mass_a,mom_a,energy_a = check_properties(outputs[:,0,:,:-1],vol,device)

          props_before.append([mass_b,mom_b,energy_b])
          props_after.append([mass_a,mom_a,energy_a])

          mass_loss = abs((mass_a - mass_b)/mass_b)
          mom_loss = abs((mom_a - mom_b)/mom_b)
          energy_loss = abs((energy_a - energy_b)/energy_b)

          l2_loss = criterion(outputs, targets)

          loss = l2_loss*loss_weights[0] \
          + mass_loss*loss_weights[1] \
          + mom_loss*loss_weights[2] \
          + energy_loss*loss_weights[3]       

          loss.backward()
          if sp_flag == 0:
              optimizer.step()
          else:
              optimizer_e.step()

          running_loss += loss.item()
          if i % 1000 == 999:
              print('   [%d, %5d] loss: %.3f' %
                    (epoch + 1, end + i + 1, running_loss / 1000))
              loss_vector.append(running_loss / 1000)
              running_loss = 0.0
            
      end += i + 1
    
    cons_array = np.concatenate((np.array(props_before),np.array(props_after)),axis=1)
    
    return loss_vector, end, cons_array


def conservation_before(f,vol,num_test,sp_flag,device):
    
    mass_in = []
    momentum_in = []
    energy_in = []

    f = torch.from_numpy(f).to(device)
    vol = torch.from_numpy(vol).to(device)

    for n in range(num_test):
      
        mass, momentum, energy = check_properties(f[n,sp_flag,:,:-1],vol,device)
                
        mass_in.append(mass)         
        momentum_in.append(momentum)         
        energy_in.append(energy)         
    
    cons_in = np.array([mass_in, momentum_in, energy_in])
    
    return cons_in


def test(f_test,df_test,vol_test,net,device,cons_in,num_test):
  
    testset = DistFuncDataset(f_test, df_test, vol_test)
    
    testloader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
      
    mass_out = []
    momentum_out = []
    energy_out = []
    
    l2_error=[]
    lt1=0
    gt1=0
    with torch.no_grad():
        for i, (data, targets, vol) in enumerate(testloader):
                        
            data, targets, vol = data.to(device), targets.to(device), vol.to(device)
          
            outputs = net(data)
            outputs = outputs.to(device)
                       
            mass, momentum, energy = check_properties(outputs[:,0,:,:-1],vol,device)
                
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

    num_error = len(cons_out[0])
    cons_error = np.zeros([3,num_error])
    
    for c in range(num_error):
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