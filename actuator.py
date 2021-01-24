import constants as C
import torch
from modules import Acoustics
from models import Elastic

#============================== 
# Electrode
#============================== 
class Electrode(Elastic, Acoustics):
    def __init__(self, a_0, model, sr=44100):
        self.a = None               # Modified radius of the electrode (m)
        self.a_0 = a_0              # Radius of the electrode (m)
        self.sr = sr                # Sampling rate (Hz)
        self.k = None               # Wave number
        self.omega = None           # Angular frequency
        #Elastic.__init__(self, model)
        self.model = Elastic(model)
    
    def set_radius(self, signal):
        self.a = self.a_0 * torch.sqrt(1 + self.model().strain(self,signal))
    
    def set_wavenumber(self, omega):
        self.omega = omega / C.sr
        self.k = omega / C.c

    def get_strain(self, signal):
        return self.model().strain(self,signal)

    def get_deformation(self, signal):
        return (C.z_0/2) * (1 + self.model().strain(self,signal))

    def get_velocity(self, signal):
        return Acoustics.calc_velocity(self, signal)

    def get_pressure(self, signal):
        return Acoustics.calc_pressure(self, signal)

#============================== 
# Surround
#============================== 
class Surround(Elastic, Acoustics):
    def __init__(self, a_0, A, model, sr=44100):
        self.a = None               # Modified radius of the electrode (m)
        self.a_0 = a_0              # Radius of the electrode (m)
        self.sr = sr                # Sampling rate (Hz)
        self.A = A                  # Radius of the rim (m)
        self.k = None               # Wave number
        self.omega = None           # Angular frequency
        #Elastic.__init__(self, model)
        self.model = Elastic(model)
    
    def surr_strain(self, signal):
        num = self.a_0**2 * self.model().strain(self,signal)
        den = self.A**2 - self.a_0**2 * (1 + self.model().strain(self,signal))
        return num / den

    def set_radius(self, signal):
        self.a = self.a_0 * torch.sqrt(1 + self.model().strain(self,signal))
    
    def set_wavenumber(self, omega):
        self.omega = omega / C.sr
        self.k = omega / C.c

    def get_strain(self, signal):
        return self.model().strain(self,signal)

    def get_deformation(self, signal):
        return (C.z_0/2) * (1 + self.surr_strain(signal))

    def get_velocity(self, signal):
        return Acoustics.calc_velocity(self, signal)
    
    def get_pressure(self, signal):
        return Acoustics.calc_pressure(self, signal)

#============================== 
# Derived class from Electrode & Surround
#============================== 
class ElastomerActuator(Electrode, Surround):
    def __init__(self, a_0, A, r, sr=44100, amplitude=None, model='dielectric'):
        self.r = r                  # Distance from COM of the DEA (m)
        self.model = model

        self.E = Electrode(a_0, model, sr)
        self.S = Surround(a_0, A, model, sr)

    def pressure(self, x, omega):
        self.E.set_wavenumber(omega)
        self.E.set_radius(x)
        self.S.set_wavenumber(omega)
        self.S.set_radius(x)

        #import matplotlib.pyplot as plt
        #plt.figure()
        ##plt.plot(x[0].cpu().numpy())
        #x = x - torch.mean(x, dim=1).unsqueeze(-1).repeat(1,x.shape[1])
        #x = x / torch.max(x, dim=1).values.unsqueeze(-1).repeat(1,x.shape[1])
        #plt.plot(x[0].cpu().numpy())
        #plt.show()

        e_cplx_prs = self.E.get_pressure(x)
        s_cplx_prs = self.S.get_pressure(x)
        out = e_cplx_prs + s_cplx_prs
        return out
    
    def deformation(self, x):
        return self.E.get_deformation(x)



