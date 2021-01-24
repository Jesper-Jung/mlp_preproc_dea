import os
import torch
import numpy as np

#============================== 
# General
#============================== 
# model \in {'dielectric', 'neohookean', 'gent', 'ArrudaBoyce', 'Ogden'}
model = 'dielectric'    # \in {'dielectric', 'neohookean', 'gent', 'ArrudaBoyce', 'Ogden'}
gpu = True
material = 'Sylgard186'  # \in {'Sylgard186', 'VHB4910'}

#============================== 
# Acoustics
#============================== 

#------------------------------ 
# Global constants
e_0 = 8.85e-12      # Free-space dielectric permittivity (F/m)
c = 343             # Speed of sound (m/s)
P_0 = 2e-5          # Reference pressure (Pa)
rho_0 = 1.21        # Equilibrium density for air (kg/m^3)
char_z = rho_0*c    # Acoustic impedance of air, at 20'C

#------------------------------ 
# Measurement
r = 1            # Distance from COM of the DEA (m)

#------------------------------ 
# Proposed
use_de = True    # use deterministic differential equation for DEA velocity


#============================== 
# Actuator
#============================== 

#------------------------------ 
# Mechanical Properties;
if material=='Sylgard186':
    e_r = 2.8         # Relative permittivity, 10:1 mix, 175% biaxially prestrained (Pa)
    Y = 1.2e6         # Young's Modulus, 10:1 mix, 175% biaxially prestrained (Pa)
elif material=='VHB4910':
    e_r = 4.0         # Relative permittivity, 400% biaxially prestrained; approx.
    Y = 1.84e7        # Young's Modulus, 400% biaxially prestrained (Pa)

#------------------------------ 
# Structure
a_0 = 2.00e-2     # Radius of the electrode (m)
A = 4.00e-2       # Outer radius of the surround (m)
if material=='Sylgard186':
    z_0 = 3.65e-4     # Original thickness, 0.0005/1.3689 (m)
elif material=='VHB4910':
    z_0 = 6.25e-5     # Original thickness, 0.001/16 (m)


#============================== 
# Models
#============================== 

#------------------------------ 
# Input Parameters
Vdc = 7.0000e3     # DC Bias (V)
Vpp = 1.000e3      # AC pk-pk (V)

#------------------------------ 
# Hyper-elastic model
mu = 1.5e6    # shear modulus (Pa)

# Gent model (sylgard 186), [Nov. 21, 2020]
''' constant reference : Improved electromechanical behavior in castable dielectric elastomer actuators,
Samin Akbari, Samuel Rosset, and Herbert R. Shea '''
mu_gent = 0.59 * 1e6
Jm_gent = 30.04

# Arruda-Boyce, [Nov. 17, 2020] 
''' Constant reference : 'Dielectric Elastomer Actuator for Soft Robotics Applications and Challenges', Jung-Hwan Youn et al. '''
maximum_stretch = 1.32
chain_length = maximum_stretch * maximum_stretch

# Ogden (sylgard 186), [Nov. 17, 2020], https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)
''' Constant reference : [April 2015] 'On the role of fibril mechanics in the work of separation of fibrillating interfaces', B.G. Vossena et al.'''
# 184, 186 정보 둘다 있음.
alpha = np.asarray([6.8, 0.799])
mu_ogden = np.asarray([0.00783/6.8, 0.682/0.799]) * 1000000



#============================== 
# Signal 
#============================== 
duration = .01           # signal length (sec)
sr = 384000              # sampling rate (Hz)
train_samples = int(duration*sr)      # number of samples for training set
infer_dur = 20           # signal length for inference (experiment)


#============================== 
# Train 
#============================== 
plot_path  = os.path.join('plot', model)        # Plotting root path
test_plot  = os.path.join(plot_path,'test')     # Test result plot dir
os.makedirs(test_plot, exist_ok=True) 
def save_plot(epoch):
    path = os.path.join(test_plot,str(epoch))
    os.makedirs(path, exist_ok=True) 
    return path

use_PRL = True
##                torch.nn.init.constant_(layer.weight,5.67e-2)
##                layer.bias.data.fill_(1e-4)
#-------------------- 
channels = [1,2**12,1,1]
# no relu at the end
#-------------------- 
# log/1.0
#batch_size = 10
#learning_rate = 1e-6
#lr_decay = [int(5000), 0.9]    # [step_size, gamma]
#pow_init = [0.85, 0.95]
#-------------------- 
# log/2.0
batch_size = 10
learning_rate = 1e-6
lr_decay = [int(10000), 0.99]    # [step_size, gamma]
pow_init = [0.85, 0.95]
#-------------------- 
V_inf = 5000
V_sup = 9000
epochs = 999999

def penalize(epoch):
    return 1e3*np.tanh(np.log(epoch+1e5) - 5)    # return 20*np.tanh(np.log(epoch+1e5) - 5)

cout_epoch = 1000             # frequency for printing train error
plot_epoch = 5000             # frequency for plotting loss
valid_epoch = 2*plot_epoch   # frequency for valid test
test_epoch = 50000            # frequency for test
loss_epoch = cout_epoch      # frequency for writing train loss
ckpt_epoch = valid_epoch     # frequency for saving checkpoint
n_vp = valid_epoch // plot_epoch
assert valid_epoch % plot_epoch == 0, "Set valid_epoch to be divisible by plot_epoch."
ckpt_name = 'filter'
def get_ckpt_dir(epoch, md, mat, cuda):
    if cuda:
        path = os.path.join('ckpt',md)
    else:
        path = os.path.join('ckpt/cpu',md)
    if md=='dielectric':
        path = os.path.join(path,mat)
    ckpt_path = os.path.join(path,str(epoch))
    os.makedirs(ckpt_path, exist_ok=True) 
    return ckpt_path

loss_scope = ['l1']    # loss to calculate
loss = ['l1']    # loss to backprop

#============================== 
# Test 
#============================== 
Test_FREQ = [20, 40, 60, 80, 100, 200, 1000, 3000, 4000, 5000, 8000, 10000, 20000]
CH_INF = 20          # Chirp signal frequency infimum
CH_SUP = 20000       # Chirp signal frequency supremum
check_G = False      # plot overall spectrogram 
compensate = False   # compensate voltage using preliminary function
def det_waveshaper(V):
    #return 2000*torch.log((V+2000)/1000)
    return torch.sqrt(8000*V)

DAQ_out_amp = 5      # [-5 ~ 5] Volt


#============================== 
# Others
#============================== 

#------------------------------ 
# Plotting
nfft = 2**11
log_xscale = False
dpi = 300

