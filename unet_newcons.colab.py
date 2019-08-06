#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:58:50 2019

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
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from google.colab import files
from torch.nn import functional as F
from torch.autograd import Variable


batch_size = 32
lr = 0.00005
momentum = 0.99
num_epochs = 100
percentage_train = 0.85
percentage_val = 0.0
lr_decay = 0.01
step_size = 5
loss_weights = [1,0.5,0.5,0.5]
nphi = 1
output_rate = 100
val_rate = 2000 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

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


def load_data_hdf(iphi):
  
  hf_f = h5py.File('/content/hdf5_data/hdf_f.h5','r')
  hf_df = h5py.File('/content/hdf5_data/hdf_df.h5','r')
  
  i_f = hf_f['i_f'][iphi]
  e_f = hf_f['e_f'][iphi]  
  i_df = hf_df['i_df'][iphi]
  e_df = hf_df['e_df'][iphi] 

  ind1,ind2,ind3 = i_f.shape
  
  f = np.zeros([ind2,2,ind1,ind1])
  df = np.zeros([ind2,2,ind1,ind1])
  props = np.zeros([ind2,2,2])

  for n in range(ind2):
    f[n,0,:,:-1] = i_f[:,n,:]
    f[n,1,:,:-1] = e_f[:,n,:]
    df[n,0,:,:-1] = i_df[:,n,:]
    df[n,1,:,:-1] = e_df[:,n,:]

    f[n,0,:,-1] = i_f[:,n,-1]
    f[n,1,:,-1] = e_f[:,n,-1]
    df[n,0,:,-1] = i_df[:,n,-1]
    df[n,1,:,-1] = e_df[:,n,-1]
    
  del i_f,e_f,i_df,e_df
  
  hf_cons = h5py.File('/content/hdf5_data/hdf_cons_fullvol.h5','r')
  
  #props[:,:,0] = hf_cons['f0_grid_vol_vonly'][...].transpose() 
  props[:,:,0] = hf_cons['f0_grid_vol'][...].transpose() 
  props[:,:,1] = hf_cons['f0_T_ev'][...].transpose()

  hf_stats = h5py.File('/content/hdf5_data/hdf_stats.h5','r')
    
  std_f = hf_stats['std_f'][...]
  std_df = hf_stats['std_df'][...]
  mean_f = hf_stats['mean_f'][...]
  mean_df = hf_stats['mean_df'][...]

  mean_f[:,:,-1] = mean_f[:,:,-2]
  mean_df[:,:,-1] = mean_df[:,:,-2]
  std_f[:,:,-1] = std_f[:,:,-2]
  std_df[:,:,-1] = std_df[:,:,-2]
  
  for n in range(ind2):
    f[n] = (f[n]-mean_f)/std_f
    df[n] = (df[n]-mean_df)/std_df
    
  mean_f = mean_f[np.newaxis]
  mean_df = mean_df[np.newaxis]
  std_f = std_f[np.newaxis]
  std_df = std_df[np.newaxis]
  
  for i in range(int(np.ceil(np.log(batch_size)/np.log(2)))):
    mean_f = np.concatenate((mean_f,mean_f),axis=0)
    mean_df = np.concatenate((mean_df,mean_df),axis=0)
    std_f = np.concatenate((std_f,std_f),axis=0)
    std_df = np.concatenate((std_df,std_df),axis=0)  
  
  mean_f = torch.from_numpy(mean_f).to(device).float()
  mean_df = torch.from_numpy(mean_df).to(device).float()
  std_f = torch.from_numpy(std_f).to(device).float()
  std_df = torch.from_numpy(std_df).to(device).float()
       
  return f,df,props,ind2,std_f,std_df,mean_f,mean_df


def load_cons_vars():
  
  hf_cons = h5py.File('/content/hdf5_data/hdf_cons_fullvol.h5','r')
  
  class conservation_variables():
    
    def __init__(self, hf_cons):
      self.f0_grid_vol = hf_cons['f0_grid_vol'][...]
      self.f0_dsmu = hf_cons['f0_dsmu'][...]
      self.f0_dvp = hf_cons['f0_dvp'][...]
      self.f0_t_ev = hf_cons['f0_T_ev'][...]
      self.f0_nvp = hf_cons['f0_nvp'][...]
      self.f0_nmu = hf_cons['f0_nmu'][...]
      self.ptl_mass = hf_cons['ptl_mass'][...]
      self.sml_ev2j = 1.6022e-19
      
  cons = conservation_variables(hf_cons)
  
  return cons 


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
    
    
def split_data(f,df,props,num_nodes):
    
    inds = np.arange(num_nodes)
    np.random.shuffle(inds) 
    
    num_train=int(np.floor(percentage_train*num_nodes))
    num_val=int(np.floor(percentage_val*num_nodes))
    
    train_inds = inds[:num_train]
    val_inds = inds[num_train:num_train+num_val]
    test_inds = inds[num_train+num_val:]
 
    f_train = f[train_inds]
    f_val = f[val_inds]
    f_test = f[test_inds]
        
    del f  
      
    df_train = df[train_inds]
    df_val = df[val_inds]
    df_test = df[test_inds]
    
    del df
        
    props_train = props[train_inds]
    props_val = props[val_inds]
    props_test = props[test_inds]
    
    del props
    
    trainset = DistFuncDataset(f_train, df_train, props_train)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             shuffle=True, pin_memory=True, num_workers=4)
    
    del f_train, df_train, props_train
    
    #valset = DistFuncDataset(f_val, df_val, props_val)
    
    #valloader = DataLoader(valset, batch_size=batch_size, 
    #                            shuffle=True, pin_memory=True, num_workers=4)
    
    valloader = None
    
    return trainloader, valloader, f_test, df_test, props_test


def check_properties(f_slice, vol):
    
    f_slice = f_slice.double()
       
    if len(f_slice.shape) == 2:
      nperp, npar = f_slice.shape
      nbatch = 1
    elif len(f_slice.shape) == 3:  
      nbatch,nperp,npar = f_slice.shape
          
    f0_smu_max = 4
    f0_vp_max = 4
    
    vpar = np.linspace(-f0_vp_max,f0_vp_max,npar) # 2*f0_nvp + 1
    vperp = np.linspace(0,f0_smu_max,nperp) # f0_nmu + 1
    
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

    #mass = (torch.sum(f_slice*mass_array).float())/nbatch
    #momentum = (torch.sum(f_slice*mom_array).float())/nbatch
    #energy = (torch.sum(f_slice*energy_array).float())/nbatch
    
    mass = torch.sum(f_slice*mass_array, dim = (1,2))
    momentum = torch.sum(f_slice*mom_array, dim = (1,2))
    energy = torch.sum(f_slice*energy_array, dim = (1,2))
                
    return mass, momentum, energy


def check_properties2(df_slice, cons):
  
  nbatch = len(df_slice)
  mass = cons.ptl_mass[1]
  
  density = torch.zeros(nbatch).float().to(device)
  momentum = torch.zeros(nbatch).float().to(device)
  energy = torch.zeros(nbatch).float().to(device)
  
  for node in range(nbatch):

    for imu in range(cons.f0_nmu+1):

      if imu==0 or imu==cons.f0_nmu:
        mu_vol = 0.5
      else:
        mu_vol = 1     
      mu = (imu*cons.f0_dsmu)**2
        
      vol = cons.f0_grid_vol[1,node]*mu_vol
      en_th = cons.f0_t_ev[1,node]*cons.sml_ev2j
      vth = np.sqrt(en_th/mass)
    
      for ivp in range(-cons.f0_nvp,cons.f0_nvp+1):
        vp = ivp*cons.f0_dvp
        en = 0.5*(vp**2 + mu)
                
        den_factor = torch.from_numpy(np.array(vol)).float().to(device)
        mom_factor = torch.from_numpy(np.array(vp*vth*mass*vol)).float().to(device)
        energy_factor = torch.from_numpy(np.array(en*en_th*vol)).float().to(device)
        
        ivp2 = ivp + cons.f0_nvp

        density[node] += df_slice[node,imu,ivp2]*den_factor
        momentum[node] += df_slice[node,imu,ivp2]*mom_factor
        energy[node] += df_slice[node,imu,ivp2]*energy_factor

  return density, momentum, energy


def check_properties3(df_slice, props, cons):
  
  if len(df_slice.shape) == 2:
    nperp, npar = df_slice.shape
    nbatch = 1
  elif len(df_slice.shape) == 3:  
    nbatch,nperp,npar = df_slice.shape
  
  mass = cons.ptl_mass[1]
  
  imu = np.linspace(0,cons.f0_nmu,cons.f0_nmu+1)
  ivp = np.linspace(-cons.f0_nvp,cons.f0_nvp,2*cons.f0_nvp+1)
  
  mu_vol = torch.ones(cons.f0_nmu+1).to(device)
  mu_vol[0], mu_vol[-1] = 0.5, 0.5 # volume vector
  
  mu = torch.tensor(imu*cons.f0_dsmu).float().to(device) # perp vector
  vp = torch.tensor(ivp*cons.f0_dvp).float().to(device) # par vector
  
  ones_tensor = torch.ones(nbatch,nperp,npar).to(device)
  
  vol = torch.einsum('ijk,i,j -> ijk',ones_tensor,props[:,0],mu_vol).to(device)
  en_th = torch.einsum('ijk,i -> ijk',ones_tensor,props[:,1]).to(device)*cons.sml_ev2j
  vth = torch.sqrt(en_th/mass).to(device)
    
  mu_tensor = torch.einsum('ijk,j -> ijk',ones_tensor,mu).to(device)
  vp_tensor = torch.einsum('ijk,k -> ijk',ones_tensor,vp).to(device)
  
  en = 0.5*(vp_tensor**2 + mu_tensor**2)
  
  dens_tensor = vol.to(device)
  mom_tensor = vp_tensor*vth*mass*vol.to(device)
  energy_tensor = en*en_th*vol.to(device)
    
  density = torch.sum(df_slice*dens_tensor, dim = (1,2))
  momentum = torch.sum(df_slice*mom_tensor, dim = (1,2))
  energy = torch.sum(df_slice*energy_tensor, dim = (1,2))
    
  return density, momentum, energy

### I used these to debug just the different version of check_properties
### not sure how useful they'll be
  
# f,df,props,num_nodes,std_f,std_df,mean_f,mean_df = load_data_hdf(0)
# cons = load_cons_vars()
  
# f = torch.from_numpy(f).float().to(device)
# df = torch.from_numpy(df).float().to(device)
# props = torch.from_numpy(props).float().to(device)
  
# a,b,c = check_properties3(f[:32,0,:,:-1],props[:32,1],cons)
# a1,b1,c1 = check_properties2(f[:32,0,:,:-1],cons)
  

def train(trainloader,valloader,sp_flag,epoch,end,std_f,std_df,mean_f,mean_df,cons):
  
    props_before = []
    props_after = []
    train_loss_vector = []
    val_loss_vector = []
    
    running_loss = 0.0
    timestart = timeit.default_timer()
    for i, (data, targets, props) in enumerate(trainloader):
        timeend = timeit.default_timer()
        #print(timeend-timestart)
     
        data, targets, props = data.to(device), targets.to(device), props.to(device)      
      
        if sp_flag == 0:
            optimizer.zero_grad()
        else:
            optimizer_e.zero_grad()

        outputs = net(data)
        outputs = outputs.to(device)
        
        if val_rate == 0:         
          val_loss = validate(valloader,std_f,std_df,mean_f,mean_df)
          val_loss_vector.append(val_loss)
        
          is_best = False
          if val_loss < np.min(val_loss_vector): ## check this
            is_best = True 

          if i % 2*val_rate == 0:
            save_checkpoint({
                             'epoch': epoch+1,
                             'state_dict': net.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict(),
                             }, is_best)
                  
        if len(data) != batch_size:
          limit = len(data)
          targets_unnorm = targets*std_df[:limit,0] + mean_df[:limit,0]
          outputs_unnorm = outputs*std_df[:limit,0] + mean_df[:limit,0]
        
        else:
          targets_unnorm = targets*std_df[:,0] + mean_df[:,0]
          outputs_unnorm = outputs*std_df[:,0] + mean_df[:,0]
          
            
        #mass_b,mom_b,energy_b = check_properties(targets_unnorm[:,0,:,:-1],vol)
        #mass_a,mom_a,energy_a = check_properties(outputs_unnorm[:,0,:,:-1],vol)  
        
        mass_b,mom_b,energy_b = check_properties3(targets_unnorm[:,0,:,:-1],props,cons) 
        mass_a,mom_a,energy_a = check_properties3(outputs_unnorm[:,0,:,:-1],props,cons)
      
        props_before.append([mass_b,mom_b,energy_b])
        props_after.append([mass_a,mom_a,energy_a])
                
        nbatch = len(targets_unnorm)
        mass_loss = torch.sum(torch.abs((mass_a - mass_b)/mass_b)).float()/nbatch
        mom_loss = torch.sum(torch.abs((mom_a - mom_b)/mom_b)).float()/nbatch
        energy_loss = torch.sum(torch.abs((energy_a - energy_b)/energy_a)).float()/nbatch        
        
        l2_loss = criterion(outputs, targets)  
                   
        if epoch < 2:
          loss = l2_loss
        else:
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
        if i % output_rate == output_rate-1:
            print('   [%d, %5d] loss: %.3f' %
                  (epoch + 1, end + i + 1, running_loss / output_rate))
            print(mass_loss.item(),mom_loss.item(),energy_loss.item())
            train_loss_vector.append(running_loss / output_rate)
            running_loss = 0.0
        timestart = timeit.default_timer()  
    end += i + 1
    
    cons_array = np.concatenate((np.array(props_before),np.array(props_after)),axis=1)
    
    return train_loss_vector, val_loss_vector, cons_array, end


def validate(valloader,std_f,std_df,mean_f,mean_df):
  
  loss = 0
  
  with torch.no_grad():
    for (data, targets, vol) in valloader:
      
      data, targets, vol = data.to(device), targets.to(device), vol.to(device)
      
      outputs = net(data)
      outputs = outputs.to(device)
      
      if len(data) != batch_size:
        limit = len(data)
        targets_unnorm = targets*std_df[:limit,0] + mean_df[:limit,0]
        outputs_unnorm = outputs*std_df[:limit,0] + mean_df[:limit,0]

      else:
        targets_unnorm = targets*std_df[:,0] + mean_df[:,0]
        outputs_unnorm = outputs*std_df[:,0] + mean_df[:,0]
        
      mass_b,mom_b,energy_b = check_properties(targets_unnorm[:,0,:,:-1],vol)
      mass_a,mom_a,energy_a = check_properties(outputs_unnorm[:,0,:,:-1],vol)
      
      nbatch = len(targets_unnorm)
      mass_loss = torch.sum(torch.abs((mass_a - mass_b)/mass_b)).float()/nbatch/1e9
      mom_loss = torch.sum(torch.abs((mom_a - mom_b)/mom_b)).float()/nbatch/1e8
      energy_loss = torch.sum(torch.abs((energy_a - energy_b)/energy_b)).float()/nbatch/1e8
      
      l2_loss = criterion(outputs, targets)
      
      loss += l2_loss*loss_weights[0] \
            + mass_loss*loss_weights[1] \
            + mom_loss*loss_weights[2] \
            + energy_loss*loss_weights[3]      

      return loss
      
  
def test(f_test,df_test,vol_test):
  
    testset = DistFuncDataset(f_test, df_test, vol_test)
    
    testloader = DataLoader(testset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
      
    props_before = []
    props_after = []
    
    l2_error=[]
    lt1=0
    gt1=0
    with torch.no_grad():
        for (data, targets, vol) in testloader:

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
    
    cons_error[0] = torch.abs((cons_array[:,3]-cons_array[:,0])/cons_array[:,0])
    cons_error[1] = torch.abs((cons_array[:,4]-cons_array[:,1])/cons_array[:,1])
    cons_error[2] = torch.abs((cons_array[:,5]-cons_array[:,2])/cons_array[:,2])
    
    print('Finished testing')
    print('Percentage with MSE<1: %d %%' % (
            100 * lt1/(lt1+gt1)))
    print('Percent error in conservation properties:\nmass: \
            %d %%\nmomentum: %d %%\nenergy: %d %%' % ( 
            100*cons_error[0].max(), 
            100*cons_error[1].max(), 
            100*cons_error[2].max()))
    
    return l2_error, cons_error


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'): 
  torch.save(state,filename)
  if is_best:
    shutil.copy(filename, 'model_best.pth.tar')
    

if __name__ == "__main__":
    
    start = timeit.default_timer()
    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=lr_decay)
    
    train_loss = []
    val_loss = []
    
    for epoch in range(num_epochs):
      print('Epoch: {}'.format(epoch+1)) 
      
      epoch1 = timeit.default_timer() 
      end = 0
      for iphi in range(nphi):
    
        print('Beginning training iphi = {}'.format(iphi))
        print('   Loading data')
        load1 = timeit.default_timer()
        f,df,props,num_nodes,std_f,std_df,mean_f,mean_df = load_data_hdf(iphi)
        cons = load_cons_vars()
        load2 = timeit.default_timer()
        print('      Loading time: {}s'.format(load2-load1))
    
        print('   Creating training set')
        trainloader,valloader,f_test,df_test,props_test = split_data(f,df[:,0,:,:],props[:,1],num_nodes)
        del f,df,props
        
        train1 = timeit.default_timer()
        ### gather testing data
        if epoch == 0:
          if iphi == 0:
            f_all_test,df_all_test,props_all_test = f_test,df_test,props_test
            del f_test,df_test,props_test
    
            print('   Starting training')
            train_loss, val_loss, cons_array, end = train(trainloader,valloader,0,epoch,end,\
                                                          std_f,std_df,mean_f,mean_df,cons)
    
          else:
            f_all_test = np.vstack((f_all_test,f_test))
            df_all_test = np.vstack((df_all_test,df_test))
            props_all_test = np.vstack((props_all_test,props_test))
            del f_test,df_test,props_test
    
            print('   Starting training')
            train_loss_to_app, val_loss_to_app, cons_to_cat, end = train(trainloader,valloader,0,epoch,end,\
                                                                         std_f,std_df,mean_f,mean_df,cons)
    
            for loss1 in train_loss_to_app:
              train_loss.append(loss1)
            for loss2 in val_loss_to_app:
              val_loss.append(loss2)
            cons_array = np.concatenate((cons_array, cons_to_cat), axis=1)
        
        else:
          del f_test,df_test,props_test
          print('   Starting training')
          train_loss_to_app, val_loss_to_app, cons_to_cat, end = train(trainloader,valloader,0,epoch,end,\
                                                                       std_f,std_df,mean_f,mean_df,cons)
          
          for loss1 in train_loss_to_app:
              train_loss.append(loss1)          
          for loss2 in val_loss_to_app:
              val_loss.append(loss2)      
          cons_array = np.concatenate((cons_array, cons_to_cat), axis=1)
             
        train2 = timeit.default_timer()
        print('Finished tranining iphi = {}'.format(iphi))
        print('   Training time for iphi = {}: {}s'.format(iphi,train2-train1))
      
      train_iterations = np.linspace(1,len(train_loss),len(train_loss))
      val_iterations = np.linspace(1,len(train_loss),len(val_loss))
      
      plt.plot(train_iterations,train_loss,'o',color='blue')
      plt.plot(val_iterations,val_loss,'o',color='orange')
    
      
      epoch2 = timeit.default_timer()
      scheduler.step()
      print('Epoch time: {}s\n'.format(epoch2-epoch1))
    
    print('Starting testing')
    l2_error_i, cons_error_i = test(f_all_test,df_all_test,props_all_test)
    print('Finished testing')
    
    stop = timeit.default_timer()
    print('Runtime: ' + str((stop-start)/3600) + 'hrs')