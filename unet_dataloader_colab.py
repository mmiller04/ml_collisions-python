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
from torch.utils.data import Dataset, DataLoader, Subset
from scipy import stats
from google.colab import files

batch_size = 32
lr = 0.0005
momentum = 0.99
num_epochs = 1
percentage_train = 0.85
lr_decay = 0.01
step_size = 1
loss_weights = [1,1,1,1]
nsp = 2

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

net = Unet().to(device)
#net_i = Unet().to(device)


class DistFuncDataset(Dataset):
    
    def __init__(self, file1, file2, file3, file4):
      
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        
        nphi,nperp,ngrid,npar = h5py.File(self.file1,'r').get('i_f').shape
        
        self.ngrid = ngrid
        self.nperp = nperp
        self.nphi = nphi
        self.npar = npar
        
    def __len__(self):
        return self.ngrid*self.nphi
     
    def __getitem__(self, index):
        time1 = timeit.default_timer()
        
        hf_f = h5py.File(self.file1,'r')
        hf_df = h5py.File(self.file2,'r')
        hf_vol = h5py.File(self.file3,'r')
        hf_stats = h5py.File(self.file4,'r')
        
        iphi = index//self.ngrid         
        igrid = index - iphi*self.ngrid
                
        xi = hf_f['i_f'][iphi,:,igrid,:]
        xe = hf_f['e_f'][iphi,:,igrid,:]
        
        time2 = timeit.default_timer()
        y_part = hf_df['i_df'][iphi,:,igrid,:]
        
        time3 = timeit.default_timer()
        x = np.empty([nsp,self.nperp,self.nperp])
        y = np.empty([1,self.nperp,self.nperp])
        
        x[0,:,:-1] = xi
        x[0,:,-1] = xi[:,-1]
             
        x[1,:,:-1] = xe
        x[1,:,-1] = xe[:,-1]
        
        y[:,:,:-1] = y_part
        y[:,:,-1] = y_part[:,-1]
        
        mean_f = hf_stats['mean_f'][...]
        mean_df = hf_stats['mean_df'][...]
        std_f = hf_stats['std_f'][...]
        std_df = hf_stats['std_df'][...]
        
        x = (x - mean_f)/std_f
        y = (y - mean_df[0])/std_df[0]
        
        time4 = timeit.default_timer()
        z = hf_vol['voli'][iphi,:,igrid]
        
        #print(time2-time1,time3-time1,time4-time1)
        
        hf_f.close()
        hf_df.close()
        hf_vol.close()
            
        return x, y, z
    

def load_data():
  
  file1 = '/content/hdf5_data/hdf_f.h5'
  file2 = '/content/hdf5_data/hdf_df.h5'
  file3 = '/content/hdf5_data/hdf_vol.h5'
  
  dataset = DistFuncDataset(file1, file2, file3)

  return dataset


def split_data(dataset, percentage_train):
  
  inds = np.arange(len(dataset))
  np.random.shuffle(inds) 
  
  num_train = int(np.floor(len(dataset)*percentage_train))
  
  train_inds = inds[:num_train]
  test_inds = inds[num_train:]
  
  trainset = Subset(dataset, train_inds)
  testset = Subset(dataset, test_inds)
  
  trainloader = DataLoader(trainset, batch_size=batch_size,
                                shuffle=True, pin_memory=True, num_workers=8)
  
  testloader = DataLoader(testset, batch_size=batch_size,
                               shuffle=True, pin_memory=True, num_workers=8)

  return trainloader, testloader


def check_properties(f_slice, vol):
    
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


def train(trainloader,sp_flag):
  
    props_before = []
    props_after = []
    loss_vector = []
    
    for epoch in range(num_epochs):
      print('Epoch: {}'.format(epoch+1)) 
      epoch1 = timeit.default_timer() 
      
      running_loss = 0.0
      timestart = timeit.default_timer()
      for i, (data, targets, vol) in enumerate(trainloader):
          timeend = timeit.default_timer()
          print(timeend-timestart)
          print('before net')
          print(i)

          data, targets, vol = data.to(device), targets.to(device), vol.to(device)
          data, targets = data.float(), targets.float()
          
          if sp_flag == 0:
              optimizer.zero_grad()
          else:
              optimizer_e.zero_grad()

          outputs = net(data)
          outputs = outputs.to(device)
          print('after net')
          mass_b,mom_b,energy_b = check_properties(data[:,0,:,:-1],vol)
          mass_a,mom_a,energy_a = check_properties(outputs[:,0,:,:-1],vol)

          props_before.append([mass_b,mom_b,energy_b])
          props_after.append([mass_a,mom_a,energy_a])

          mass_loss = abs((mass_a - mass_b)/mass_b)
          mom_loss = abs((mom_a - mom_b)/mom_b)
          energy_loss = abs((energy_a - energy_b)/energy_b)
          print('after conservation')
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
          print('after backprop')
          running_loss += loss.item()
          #if i % 100 == 99:
          #    print('   [%d, %5d] loss: %.3f' %
          #          (epoch + 1, i + 1, running_loss / 100))
          #    loss_vector.append(running_loss / 100)
          #    running_loss = 0.0
          timestart = timeit.default_timer()
      epoch2 = timeit.default_timer()
      print('Time for epoch {}: {}s'.format(epoch,epoch2-epoch1))
      
    cons_array = np.concatenate((np.array(props_before),np.array(props_after)),axis=1)
    
    return loss_vector, cons_array


def test(testloader):
  
    props_before = []
    props_after = []
    
    l2_error=[]
    lt1=0
    gt1=0
    with torch.no_grad():
        for i, (data, targets, vol) in enumerate(testloader):
                        
            data, targets, vol = data.to(device), targets.to(device), vol.to(device)
          
            outputs = net(data)
            outputs = outputs.to(device)
            
            props_before.append(check_properties(data[:,0,:,:-1],vol))          
            props_after.append(check_properties(outputs[:,0,:,:-1],vol))
            
            loss = criterion(outputs, targets)
            l2_error.append(loss.item()*100)
            if loss.item()*100 < 1:
                lt1+=1
            else:
                gt1+=1
    
    cons_array = np.concatenate((np.array(props_before),np.array(props_after)),axis=1)

    num_error = len(cons_array)
    cons_error = np.zeros([3,num_error])
    
    cons_error[0] = (cons_array[:,3]-cons_array[:,0])/cons_array[:,0]
    cons_error[1] = (cons_array[:,4]-cons_array[:,1])/cons_array[:,1]
    cons_error[2] = (cons_array[:,5]-cons_array[:,2])/cons_array[:,2]

    print('Percentage with MSE<1: %d %%' % (
            100 * lt1/(lt1+gt1)))
    print('Percent error in conservation properties:\nmass: \
            %d %%\nmomentum: %d %%\nenergy: %d %%' % ( 
            100*cons_error[0,:].max(), 
            100*cons_error[1,:].max(), 
            100*cons_error[2,:].max()))
    
    return l2_error, cons_error


if __name__  == "__main__":
    
    start = timeit.default_timer()

    print('Loading data')
    load1 = timeit.default_timer()
    dataset = load_data()
    trainloader, testloader = split_data(dataset, percentage_train)
    load2 = timeit.default_timer()
    print('Finished loading - total loading time: {}s'.format(load2-load1))
    
    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=lr_decay)
    
    print('Starting training')
    train1 = timeit.default_timer()
    #%lprun -f DistFuncDataset.__getitem__ loss_vector, cons_train = train(trainloader,0)
    loss_vector, cons_train = train(trainloader,0)
    train2 = timeit.default_timer()
    print('Finished training - total training time = {} hrs'.format((train2-train1)/3600))
    
    iterations = np.linspace(1,len(loss_vector),len(loss_vector))
    plt.plot(iterations,loss_vector)
    
    print('Starting testing')
    test1 = timeit.default.timer()
    l2_error, cons_error, cons_test = test(testloader)
    test2 = timeit.default.timer()
    print('Finished testing - total testing time = {}s'.format(test2-test1))
    
    stop = timeit.default_timer()
    print('Runtime: ' + str((stop-start)/3600) + 'hrs')