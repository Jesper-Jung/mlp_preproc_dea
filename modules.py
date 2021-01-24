import constants as C
import torch
import torch.nn.functional as F
from utils import cplx_mult

#============================== 
# Acoustics module
#============================== 
class Acoustics:
    def calc_velocity(self, signal):
        # TODO: this should be re-formulated
        if C.use_de:
            strain = self.get_strain(signal)
            omg = torch.cumsum(self.omega, dim=1)
            #omg = self.omega
            #------------------------------ 
            p1 = -1 * C.e_0*C.e_r / (C.Y*C.z_0)
            p2 = signal / (3*strain**2 + 4*strain + 1)
            p3 = self.omega * C.Vpp * torch.cos(omg)
            out = p1 * p2 * p3

            #D = self.get_deformation(signal).unsqueeze(0)
            #D_ = F.pad(D[:,...,1:], (0,1), 'reflect')
            #dD = (D_ - D).squeeze()
            #dD[:,-1] = dD[:,-2]
            #dt = 1 / self.sr
            #out_delta = (1/self.omega) * (dD / dt)
            #print(torch.mean(out - out_delta))
        else:
            D = self.get_deformation(signal).unsqueeze(0)
            D_ = F.pad(D[:,...,1:], (0,1), 'reflect')
            dD = (D_ - D).squeeze()
            dD[:,-1] = dD[:,-2]
            dt = 1 / self.sr
            out = (1/self.omega) * (dD / dt)

        return out
    
    def calc_pressure(self, signal):
        #integral = torch.cos(self.k * C.r) \
        #         - torch.cos(self.k * torch.sqrt(self.a**2 + C.r**2))
        #return C.char_z * self.calc_velocity(signal) * integral
        return (.5 * C.char_z) * self.calc_velocity(signal) * (self.a / C.r) * self.k * self.a







