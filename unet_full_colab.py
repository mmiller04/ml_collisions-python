#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:38:19 2019

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
import statistics
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from google.colab import files

batch_size = 32
lr = 0.00005
momentum = 0.99
num_epochs = 10
percentage_train = 0.85
lr_decay = 0.01
step_size = 2
loss_weights = [1,0.5,0.5,0.5]
nphi = 4

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


def load_data_hdf(iphi):
  
  hf_f = h5py.File('/content/hdf5_data/hdf_f.h5','r')
  hf_df = h5py.File('/content/hdf5_data/hdf_df.h5','r')
  hf_vol = h5py.File('/content/hdf5_data/hdf_vol.h5','r')
  
  i_f = hf_f['i_f'][iphi]
  e_f = hf_f['e_f'][iphi]  
  i_df = hf_df['i_df'][iphi]
  e_df = hf_df['e_df'][iphi] 
  i_vol = hf_vol['voli'][iphi]
  e_vol = hf_vol['vole'][iphi]

  ind1,ind2,ind3 = i_f.shape
  
  f = np.zeros([ind2,2,ind1,ind1])
  df = np.zeros([ind2,2,ind1,ind1])
  vol = np.zeros([ind2,2,ind1])

  for n in range(ind2):
    f[n,0,:,:-1] = i_f[:,n,:]
    f[n,1,:,:-1] = e_f[:,n,:]
    df[n,0,:,:-1] = i_df[:,n,:]
    df[n,1,:,:-1] = e_df[:,n,:]

    f[n,0,:,-1]=i_f[:,n,-1]
    f[n,1,:,-1]=e_f[:,n,-1]
    df[n,0,:,-1]=i_df[:,n,-1]
    df[n,1,:,-1]=e_df[:,n,-1]
    
    vol[n,0,:] = i_vol[:,n]
    vol[n,1,:] = e_vol[:,n]

  del i_f,e_f,i_df,e_df,i_vol,e_vol
    
  std_f = np.empty([batch_size,2,ind1,ind1])
  std_df = np.empty([batch_size,2,ind1,ind1])
  mean_f = np.empty([batch_size,2,ind1,ind1])
  mean_df = np.empty([batch_size,2,ind1,ind1])

  for i in range(ind1):
    for j in range(ind1):
      f[:,0,i,j] = stats.zscore(f[:,0,i,j])
      f[:,1,i,j] = stats.zscore(f[:,1,i,j])
      df[:,0,i,j] = stats.zscore(df[:,0,i,j])
      df[:,1,i,j] = stats.zscore(df[:,1,i,j])

      std_f[:,0,i,j] = np.std(f[:,0,i,j])
      std_f[:,1,i,j] = np.std(f[:,1,i,j])    
      std_df[:,0,i,j] = np.std(df[:,0,i,j])
      std_df[:,1,i,j] = np.std(df[:,1,i,j])

      mean_f[:,0,i,j] = np.mean(f[:,0,i,j])
      mean_f[:,1,i,j] = np.mean(f[:,1,i,j])    
      mean_df[:,0,i,j] = np.mean(df[:,0,i,j])
      mean_df[:,1,i,j] = np.mean(df[:,1,i,j])

  std_f = torch.from_numpy(std_f).to(device).float()
  std_df = torch.from_numpy(std_df).to(device).float()
  mean_f = torch.from_numpy(mean_f).to(device).float()
  mean_df = torch.from_numpy(mean_df).to(device).float()
    
  return f,df,vol,ind2,std_f,std_df,mean_f,mean_df


class DistFuncDataset(Dataset):

    def __init__(self, f_array, df_array, vol_array):
        self.data = torch.from_numpy(f_array).float()
        self.target = torch.from_numpy(df_array).float()
        self.vol = torch.from_numpy(vol_array).float()
        
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        y = y.view(-1,32,32)
        z = self.vol[index]
            
        return x, y, z
    
    
def split_data(f,df,vol,num_nodes):
    
    # shuffle data up
    #expand_df = np.expand_dims(df,axis=1)
    #expand_vol = np.expand_dims(vol,axis=1)
    #expand_vol = np.expand_dims(expand_vol,axis=3)
    
    #vol_ones = np.ones((num_nodes,1,len(vol[0,:]),len(vol[0,:])))
    #expand_vol = vol_ones*expand_vol
    
    #del vol_ones
    
    #all_data = np.concatenate((f,expand_df,expand_vol),axis=1)
    #np.random.shuffle(all_data)
    
    #f = all_data[:,:2,:,:]
    #df = all_data[:,2,:,:]
    #vol = all_data[:,3,:,0]
    
    #del all_data
    
    num_train=int(np.floor(percentage_train*num_nodes))
    num_test = num_nodes - num_train
    
    f_train = f[:num_train,:,:,:]
    f_test = f[num_train:,:,:,:]
        
    del f  
      
    df_train = df[:num_train,:,:]
    df_test = df[num_train:,:,:]
    
    del df
    
    vol_train = vol[:num_train,:]
    vol_test = vol[num_train:,:]
    
    del vol
    
    trainset = DistFuncDataset(f_train, df_train, vol_train)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, pin_memory=True, num_workers=4)
    
    del f_train, df_train, vol_train
    
    return trainloader, f_test, df_test, vol_test, num_train, num_test


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


def train(trainloader,net,device,sp_flag,loss_vector,epoch,iphi,end,std_f,std_df,mean_f,mean_df):
  
    mass_before=[]
    mass_after=[]
    mom_before=[]
    mom_after=[]
    energy_before=[]
    energy_after=[]
  
    running_loss = 0.0
    timestart = timeit.default_timer()
    for i, (data, targets, vol) in enumerate(trainloader):
        timeend = timeit.default_timer()
        #print(timeend-timestart)
     
        data, targets, vol = data.to(device), targets.to(device), vol.to(device)

        if sp_flag == 0:
            optimizer.zero_grad()
        else:
            optimizer_e.zero_grad()

        outputs = net(data)
        outputs = outputs.to(device)
        
        if len(data) != batch_size:
          limit = len(data)
          data = data*std_f[:limit] + mean_f[:limit]
          outputs = outputs*std_df[:limit,0] + mean_df[:limit,0]
        
        else:
          data = data*std_f + mean_f
          outputs = outputs*std_df[:,0] + mean_df[:,0]

        mass_b,mom_b,energy_b = check_properties(data[:,0,:,:-1],vol)
        mass_a,mom_a,energy_a = check_properties(outputs[:,0,:,:-1],vol)
        
        mass_before.append(mass_b)
        mass_after.append(mass_a)
        mom_before.append(mom_b)
        mom_after.append(mom_a)                
        energy_before.append(energy_b)
        energy_after.append(energy_a)

        mass_loss = np.abs((mass_a - mass_b)/mass_b)
        mom_loss = np.abs((mom_a - mom_b)/mom_b)
        energy_loss = np.abs((energy_a - energy_b)/energy_b)

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
        
        timestart = timeit.default_timer()  
    end += i + 1
    
    cons_array = np.array([mass_before,mass_after,mom_before,mom_after,energy_before,energy_after])
    
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
            print(i)            
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
        cons_error[0,c] += np.abs((cons_out[0,c] - cons_in[0,c])/cons_in[0,c])
        cons_error[1,c] += np.abs((cons_out[1,c] - cons_in[1,c])/cons_in[1,c])
        cons_error[2,c] += np.abs((cons_out[2,c] - cons_in[2,c])/cons_in[2,c])
    
    print('Finished testing')
    print('Percentage with MSE<1: %d %%' % (
            100 * lt1/(lt1+gt1)))
    print('Percent error in conservation properties:\nmass: \
            %d %%\nmomentum: %d %%\nenergy: %d %%' % ( 
            100*cons_error[0,:].max(), 
            100*cons_error[1,:].max(), 
            100*cons_error[2,:].max()))
    
    return l2_error, cons_error


if __name__ == "__main__":
    
    start = timeit.default_timer()
    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(net_i.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=lr_decay)
    
    loss_vector=[]
    for epoch in range(num_epochs):
      print('Epoch: {}'.format(epoch+1)) 
      
      epoch1 = timeit.default_timer() 
      end = 0
      for iphi in range(nphi):
    
        print('Beginning training iphi = {}'.format(iphi))
        print('   Loading data')
        load1 = timeit.default_timer()
        f,df,vol,num_nodes,std_f,std_df,mean_f,mean_df = load_data_hdf(iphi)
        load2 = timeit.default_timer()
        print('      Loading time: {}s'.format(load2-load1))
    
        print('   Creating training set')
        trainloader,f_test,df_test,vol_test,num_train,num_test = split_data(f,df[:,0,:,:],vol[:,0,:],num_nodes)
        del f,df,vol
        
        train1 = timeit.default_timer()
        ### gather testing data
        if epoch == 0:
          if iphi == 0:
            f_all_test,df_all_test,vol_all_test = f_test,df_test,vol_test
            del f_test,df_test,vol_test
    
            print('   Starting training')
            loss_vector, end, cons_array = train(trainloader,net_i,device,0,loss_vector,epoch,iphi,end,\
                                                std_f,std_df,mean_f,mean_df)
    
          else:
            f_all_test = np.vstack((f_all_test,f_test))
            df_all_test = np.vstack((df_all_test,df_test))
            vol_all_test = np.vstack((vol_all_test,vol_test))
            del f_test,df_test,vol_test
    
            print('   Starting training')
            loss_vector, end, cons_to_cat = train(trainloader,net_i,device,0,loss_vector,epoch,iphi,end,\
                                                 std_f,std_df,mean_f,mean_df)
    
            cons_array = np.concatenate((cons_array, cons_to_cat), axis=1)
        
        else:
          del f_test,df_test,vol_test
          print('   Starting training')
          loss_vector, end, cons_to_cat = train(trainloader,net_i,device,0,loss_vector,epoch,iphi,end,\
                                               std_f,std_df,mean_f,mean_df)
    
          cons_array = np.concatenate((cons_array, cons_to_cat), axis=1)
        
        train2 = timeit.default_timer()
        print('Finished tranining iphi = {}'.format(iphi))
        print('   Training time for iphi = {}: {}s'.format(iphi,train2-train1))
        
      epoch2 = timeit.default_timer()
      scheduler.step()
      print('Epoch time: {}s\n'.format(epoch2-epoch1))
    
    iterations=np.linspace(1,len(loss_vector),len(loss_vector))
    plt.plot(iterations,loss_vector)
    
    #print('Calculating conservation properties')
    #cons_in = conservation_before(f_all_test,vol_all_test,num_test,0,device)
    #print('Starting testing')
    #l2_error_i, cons_error_i = test(f_all_test,df_all_test,vol_all_test,net_i,device,consi_in,num_test)
    #print('Finished testing')
    
    #stop = timeit.default_timer()
    #print('Runtime: ' + str((stop-start)/3600) + 'hrs')