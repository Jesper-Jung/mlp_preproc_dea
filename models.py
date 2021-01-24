import constants as C
import torch
import numpy as np

### NOTICE [Nov. 17, 2020]
### NOTE : Ogden model이 initial condition 때문에 stretch solution 구할 때 조금 불안정함. => please change initial condition to adequate.
### NOTE : Ogden model이 initial condition 때문에 stretch solution 구할 때 조금 불안정함. => please change initial condition to adequate.
### NOTE : Ogden model이 initial condition 때문에 stretch solution 구할 때 조금 불안정함. => please change initial condition to adequate.
### NOTE : Ogden model이 initial condition 때문에 stretch solution 구할 때 조금 불안정함. => please change initial condition to adequate.
### NOTE : Ogden model이 initial condition 때문에 stretch solution 구할 때 조금 불안정함. => please change initial condition to adequate.


#============================== 
# Dielectric model
#============================== 
class Dielectric:
    def strain(self, signal):
        """
        Calculate the (true) strain of DEA.
        """
        epsilon = (-1/C.Y) * C.e_0 * C.e_r * (signal / C.z_0)**2
        K = 2 + 27*epsilon.cuda()
        K = K.type(torch.cfloat)
        #print(K)
        S = (K*K - 4)**.5
        E = (.5 * (K + S))**(1/3)
        #------------------------------ 
        out = -1*(2/3) + (1/3)*(E + 1/E)    # is almost real
        #print("out",out)
        return out.real
        #------------------------------ 

    def get_stress_strain(self, V, true_strain=True):
        print(".. Dielectric s-s")
        stress = C.e_0 * C.e_r * ((V / C.z_0)**2)
        epsilon = (-1/C.Y)*stress
        K = 2 + 27*epsilon
        E = (.5 * (K + torch.pow(K*K - 4, .5)))**(1/3)
        #E = (.5 * (K + torch.sqrt(K*K - 4)))**(1/3)
        strain = 1/3 - (1/3)*(o1*E + o2/E)        # is engineering strain
        if true_strain:
            strain -= 1
        return stress, strain

#============================== 
# Neo-hookean model
#============================== 
class NeoHookean:
    """
    Incompressible neo-Hookean material under uniaxial compression.
    """
    def Maxwell_stress(self, V):
        return C.e_0 * C.e_r * ((V / C.z_0)**2)
    
    def strain(self, signal):
        stress = self.Maxwell_stress(signal)
        strain = self.calc_strain(stress / C.mu)    # is NOT engineering strain
        return strain
    
    def calc_strain(self, x):
        """
        x: stress
        return (real) solution of
            x = 2 * C1 * (y^2 - y^{-1})
        for the variable y.
        """
        C1 = C.mu / 2    # material constant
        num = (2/3)**(1/3) * x
        den = ((3**.5) * (27*((2*C1)**6)-4*((2*C1)**3)*(x**3))**.5 - 9*(2*C1)**3)**(1/3)
        s1 = num/den
        num = den
        den = (2**(1/3)) * (3**(2/3)) * 2*C1
        s2 = num/den
        return s1 + s2
    
    def get_stress_strain(self, V1, V2, points, true_strain=True):
        print(".. Neo-hookean s-s")
        sg1 = self.Maxwell_stress(V1)
        sg2 = self.Maxwell_stress(V2)
        stress = torch.linspace(sg1, sg2, points)
        strain = self.calc_strain(stress / C.mu)
        if not true_strain:
            strain += 1
        return stress, strain
    
    def test_stress_strain(self, points, true_strain=False):
        lam = torch.linspace(1, 1.16, points)
        #lam = torch.linspace(.5, 2, points)
        stress = C.mu*(lam**2 - 1/lam)
        strain = lam-1 if true_strain else lam
        return stress, strain


#============================== 
# Gent model [Dec. 23, 2020]
#============================== 
class Gent:
    """
    Incompressible gent material under uniaxial compression.
    """
    def Maxwell_stress(self, V):
        return C.e_0 * C.e_r * ((V / C.z_0)**2)
    
    def strain(self, signal):
        _calculation = gent_calc.apply
        stress = self.Maxwell_stress(signal)
        strain = _calculation(stress) - 1    # Engineering stress <-> Engineering stretch (strain)
        return strain
  
class gent_calc(torch.autograd.Function):
    '''
    calculate model expression of Gent and backward gradient.

    INPUT
    -----
    * input: tensor, (B, T)
        (Batched) sequence of engineer stress.

    OUTPUT
    ------
    * output: tensor, (B, T)
        (Batched) Sequence of engineer stretch.
    '''
    @staticmethod
    def forward(ctx, input):
        if len(input.shape) > 1:
            seq_len = input.shape[-1]
        else:
            seq_len = input.shape[0]
        ctx.seq_len = seq_len

        input_numpy = input.contiguous().view(-1).cpu().detach().numpy()
        result = torch.tensor(calc_stretch_GN(input_numpy))
        
        ctx.save_for_backward(result)
        return result.contiguous().view(-1, seq_len)

    @staticmethod
    def backward(ctx, grad_output):
        stretch, = ctx.saved_tensors
        stretch_numpy = stretch.cpu().detach().numpy()
        gradient = calc_gradient_of_stretch_GN(stretch_numpy)

        go_numpy = grad_output.contiguous().view(-1).cpu().detach().numpy()
        grad_input = gradient * go_numpy

        return torch.from_numpy(grad_input).contiguous().view(-1, ctx.seq_len)

def calc_gradient_of_stretch_GN(stretch):
    _LeftCauchy = np.power(stretch, 2) + 2/stretch

    left_term = (1 + np.power(stretch, -3)) / (C.Jm_gent + 3 - _LeftCauchy) * C.mu * C.Jm_gent
    right_term = 2 * np.power((stretch - np.power(stretch, -2)), 2) / np.power((C.Jm_gent + 3 - _LeftCauchy), 2) * C.mu * C.Jm_gent

    return 1 / (left_term + right_term)

def calc_stretch_GN(stress):
    """
    ### Engineering strain for input -> Engineering stretch for output ###
    
    x: stress
    return (real) solution of
        x = ((mu Jm) / (Jm - I1 + 3)) * (y^2 - y^{-1})
    for the variable y, where I1 = y^2 + 2/y (맞나?)
    """
    def __left_poly():
        ''' denoted coefficients by descending '''
        __left_polynomial = [0, 1, 0, 0, -1]
        __left_polynomial = C.mu_gent * C.Jm_gent * np.asarray(__left_polynomial)
        return __left_polynomial

    def __right_poly():
        ''' denoted coefficients by descending '''
        __right_polynomial = [-1, 0, C.Jm_gent + 3, -2, 0]
        __right_polynomial = np.asarray(__right_polynomial)
        return __right_polynomial

    # setting polynomial equations
    __left_polybulk = np.repeat(__left_poly()[:, np.newaxis], len(stress), axis=1)
    __right_polybulk = np.repeat(__right_poly()[:, np.newaxis], len(stress), axis=1) * stress

    # transposition (sum up)
    polynomial_bulk = __left_polybulk - __right_polybulk

    # solve the equations
    solution_stretch = np.empty_like(stress)
    for i in range(len(stress)):
        # solving
        __solution = np.roots(polynomial_bulk[:, i])

        __solution = __solution[np.abs(np.imag(__solution))<1e-10] # non-imag
        __solution_stretch = __solution[np.real(__solution)>0] # pos,

        if len(__solution_stretch) is 2:
            # real-solution can appear up to 2 solutions in negative stress, s.t. lambda<1 and lambda>1
            index = np.argmin(np.real(__solution_stretch))
            __solution_stretch = __solution_stretch[index, np.newaxis]

        assert len(__solution_stretch) == 1, "해가 이상해" # check real solution is only one
        solution_stretch[i] = np.real(__solution_stretch)[0]

    return solution_stretch


#============================== 
# Arruda-Boyce model [Dec 23, 2020]
#============================== 
class ArrudaBoyce:
    """
    Incompressible Arruda Boyce material under uniaxial compression.
    """
    def Maxwell_stress(self, signal):
        return C.e_0 * C.e_r * (signal/C.z_0)**2
    
    def strain(self, signal):
        stress = self.Maxwell_stress(signal)
        stretch = self.calc_strain(stress)
        return stretch - 1 
 

class arrudaboyce_calc(torch.autograd.Function):
    '''
    calculate model expression of Arruda Boyce and backward gradient.

    INPUT
    -----
    * input: tensor, (B, T)
        (Batched) sequence of engineer stress.

    OUTPUT
    ------
    * output: tensor, (B, T)
        (Batched) Sequence of engineer stretch.
    '''
    @staticmethod
    def forward(ctx, input):
        if len(input.shape) > 1:
            seq_len = input.shape[-1]
        else:
            seq_len = input.shape[0]
        ctx.seq_len = seq_len

        input_numpy = input.contiguous().view(-1).cpu().detach().numpy()
        result = torch.tensor(calc_stretch_AB(input_numpy))
        
        ctx.save_for_backward(result)
        return result.contiguous().view(-1, seq_len)

    @staticmethod
    def backward(ctx, grad_output):
        stretch, = ctx.saved_tensors
        stretch_numpy = stretch.cpu().detach().numpy()
        gradient = calc_gradient_of_stretch_AB(stretch_numpy)

        go_numpy = grad_output.contiguous().view(-1).cpu().detach().numpy()
        grad_input = gradient * go_numpy

        return torch.from_numpy(grad_input).contiguous().view(-1, ctx.seq_len)

def calc_stretch_AB(stress):
    # calcaulate polynomial (NOTE: Descending)
    _polynomial = _calc_remain_coeff()

    # calculate material constant
    material_const = _material_const()

    # sum up (transposition right to left)
    polynomial_bulk = np.repeat(_polynomial[:, np.newaxis], len(stress), axis=1) # 16 x len(stress) matrix.
    polynomial_bulk[9] -= stress/(2*material_const)

    # calculate roots of polynomials
    stretch_solution = np.empty_like(stress)
    for i in range(len(stress)):
        _solution = np.roots(polynomial_bulk[:, i]) # one solution of each polynomial

        # extract positive-real solution (only one is available)
        _solution = _solution[np.abs(np.imag(_solution))<1e-10] # real
        _stretch_solution = _solution[np.real(_solution)>0] # positive

        assert len(_stretch_solution) == 1, "해가 이상해" # check real solution is only one

        stretch_solution[i] = np.real(_stretch_solution) # extract solution

    return stretch_solution


def _calc_remain_coeff():
    # setting coefficients of rest term (Ascending)
    rest_order5 = [16, 0, 0, 32, 0, 0, 24, 0, 0, 8, 0, 0, 1]
    rest_order5 = np.asarray(rest_order5)

    rest_order4 = [0, 8, 0, 0, 12, 0, 0, 6, 0, 0, 1, 0, 0]
    rest_order4 = np.asarray(rest_order4)

    rest_order3 = [0, 0, 4, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0]
    rest_order3 = np.asarray(rest_order3)

    rest_order2 = [0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    rest_order2 = np.asarray(rest_order2)

    rest_order1 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    rest_order1 = np.asarray(rest_order1)

    # concatenate
    rest_order = np.concatenate((rest_order1[:, np.newaxis],
                                rest_order2[:, np.newaxis],
                                rest_order3[:, np.newaxis],
                                rest_order4[:, np.newaxis],
                                rest_order5[:, np.newaxis]), axis=1) # 13x5 matrix

    # generate coefficient
    ###########################################################################################
    # NOTE: add C.chain_length in 'constant.py' later, criterioned by 10:1 mixed sylgard 184. #
    ###########################################################################################
    beta = np.power(1/C.chain_length, np.arange(5))

    alpha = [1/2, 1/20, 11/1050, 19/7000, 519/673750]
    alpha = np.asarray(alpha)

    rest_coeff = (np.arange(5) + 1) * alpha * beta # 5-vector

    # calculate rest polynomial
    rest_poly = np.matmul(rest_order, rest_coeff) # 13-vector


    # calculate all polynomial
    poly_term1 = np.concatenate((np.zeros(3), rest_poly), axis=0)
    poly_term2 = np.concatenate((rest_poly, np.zeros(3)), axis=0)
    polynomial = poly_term1 - poly_term2 # 16-vector

    return polynomial[::-1]

def _material_const():
    coeff = [1, 3/5, 99/175, 513/875, 42039/67375]
    coeff = np.asarray(coeff)

    ###########################################################################################
    # NOTE: add C.chain_length in 'constant.py' later, criterioned by 10:1 mixed sylgard 184. #
    ###########################################################################################
    beta = np.power(1/C.chain_length, np.arange(5))   

    _proportion_const = (coeff * beta).sum()

    return C.mu/_proportion_const

def calc_gradient_of_stretch_AB(stretch):
    alpha = np.asarray([1/2, 1/20, 11/1050, 19/7000, 519/673750])
    beta = np.power(1/C.chain_length, np.arange(5))

    def FirInv_LeftCauchy(stretch):
        return np.power(stretch, 2) + 2 / stretch

    _first_term = np.concatenate((
        np.power(np.expand_dims(stretch, axis=0), 0),
        np.power(np.expand_dims(stretch, axis=0), 1),
        np.power(np.expand_dims(stretch, axis=0), 2),
        np.power(np.expand_dims(stretch, axis=0), 3),
        np.power(np.expand_dims(stretch, axis=0), 4)
    ), axis=0)

    coeff_first_term = np.arange(1, 6) * alpha * beta

    first_term = (1 + 2 / np.power(stretch, 3)) * np.dot(coeff_first_term, _first_term)

    _second_term = np.concatenate((
        np.power(np.expand_dims(stretch, axis=0), 0),
        np.power(np.expand_dims(stretch, axis=0), 1),
        np.power(np.expand_dims(stretch, axis=0), 2),
        np.power(np.expand_dims(stretch, axis=0), 3),
    ), axis=0)

    coeff_second_term = np.arange(2, 6) * np.arange(1, 5) * alpha[1:] * beta[1:]

    second_term = 2 * (stretch - 1/np.power(stretch, 2)) * np.dot(coeff_second_term, _second_term)

    C1 = _material_const()
    gradients = 1/(2*C1) / (first_term + second_term)

    return gradients



#============================== 
# Ogden model [Nov. 17, 2020]
#============================== 
from scipy.optimize import newton_krylov

### NOTE : initial condition 때문에 stretch solution 구할 때 조금 불안정함.
class Ogden:
    """
    Incompressible Ogden (2-order) material under uniaxial compression.
    """
    def Maxwell_stress(self, signal):
        return C.e_0 * C.e_r * (signal/C.z_0)**2
    
    def strain(self, signal):
        stress = self.Maxwell_stress(signal)
        stretch = self.calc_strain(stress)
        return stretch - 1
 
    @staticmethod
    def calc_strain(stress, batch_size = 1000):
        '''
        Calculate strain from stress using Ogden model (2-order)
        ### not engineering stress -> not engineering strain ###

        INPUT
        -----
        x: stress(not engineering)

        OUTPUT
        ------
        y: non-linear stretch(strain) output of stress(not engineering) by the model.
        '''
        ###################################################################################################
        # NOTE: add C.mu_odgen and C.alpha in 'constant.py' later, criterioned by 10:1 mixed sylgard 184. #
        ###################################################################################################

        # define function (2-order ogden model)
        def G(batch_stress):
            '''
            ### numpy -> Function ###

            INPUT
            -----
            - batch_stress : numpy
                input stress information

            OUTPUT
            ------
            - F : Function
                2-order ogden model parameterized stretch by x
            '''
            def F(x):
                nonlocal batch_stress
                ''' function of parameter x, 2-order ogden model '''
                value1 = C.mu_odgen[0] * (np.power(x, C.alpha[0]-1) - np.power(x, -0.5*C.alpha[0]-1))
                value2 = C.mu_odgen[1] * (np.power(x, C.alpha[1]-1) - np.power(x, -0.5*C.alpha[1]-1))

                value = value1 + value2
                return value - stress
            return F    

        # rearrange for batch
        batch_number = int(len(stress)/batch_size)
        stress = stress[:batch_number * batch_size].reshape(batch_number, batch_size)

        # calculate solution for each batches
        stretch_solution = np.zeros(batch_number * batch_size)
        for i in range(batch_number):
            __stretch_solution = newton_krylov(G(stress[i]), np.ones(len(stress[i]))*3)
            stretch_solution[i*batch_size:(i+1)*batch_size] = __stretch_solution
        
        return stretch_solution
                    



class Elastic(Dielectric, NeoHookean, Gent):
    def __init__(self, model):
        if model == 'dielectric':
            self.definition = Dielectric
            print("**  Using Proposed model..")
        elif model == 'neohookean':
            self.definition = NeoHookean
            print("**  Using Neo-hookean model..")
        elif model == 'gent':
            self.definition = Gent
            print("**  Using Gent model..")
        elif model == 'ArrudaBoyce':
            self.definition = ArrudaBoyce
            print("**  Using ArrudaBoyce model..")
        elif model == 'Ogden':
            self.definition = Ogden
            print("**  Using Ogden model..")
        else:
            raise NotImplementedError
    def __call__(self):
        return self.definition


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # LaTeX Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    true_strain = False 
    V1 = 2.0e3
    V2 = 5.0e3
    V2_2 = 8e3
    pt = 100

    DI = Dielectric()
    NH = NeoHookean()
    GN = Gent()
    AB = ArrudaBoyce()
    OG = Ogden()

    V = torch.linspace(V1, V2, pt)
    plt.figure(figsize=(5,5))
    #============================== 
    # Dielectric
    stress, strain = DI.get_stress_strain(V,true_strain=true_strain)
    stress = stress.numpy()
    strain = strain.numpy()
    label = '{}, $\mu$={} (MPa)'.format('Target', C.mu*1e-6)
    plt.plot(strain, stress*1e-6, label=label)
    ##============================== 
    ## Neo-hookean
    #stress, strain = NH.test_stress_strain(points=pt,true_strain=true_strain)
    #stress = stress.numpy()
    #strain = strain.numpy()
    #label = 'Neo-hookean, from strain to stress'
    #plt.plot(strain, stress*1e-6, '.', label=label)
    ##------------------------------ 
    stress, strain = NH.get_stress_strain(V1, V2_2, points=pt,true_strain=true_strain)
    stress = stress.numpy()
    strain = strain.numpy()
    label = '{}, $\mu$={} (MPa)'.format('Neo-hookean', C.mu*1e-6)
    plt.plot(strain, stress*1e-6, label=label)
    #============================== 
    # Gent
    #------------------------------ 

    plt.grid(True)
    strain_label = '$\\varepsilon$' if true_strain else '$\lambda$'
    plt.xlabel(strain_label)
    plt.ylabel('Stress $\sigma_{11}$ (MPa)')
    plt.title('Uniaxial stretch, incompressible')
    plt.legend()
    plt.savefig("./test_model.png", dpi=500)

    #============================== 
    # Polynomial Fitting
    #============================== 

    V1 = 2.0e3
    V2 = 5.0e3
    pt = 10000
    plt.figure(figsize=(5,5))
    V = np.linspace(V1, V2, pt)
    Vt = torch.linspace(V1, V2, pt)
    TF_order = 10
    order_min = 5
    order_sup = 10
    #============================== 
    # Dielectric
    strain = DI.strain(Vt)
    #_, strain = DI.get_stress_strain(Vt,true_strain=true_strain)
    strain = strain.numpy()
    z_0 = np.polyfit(strain, V, TF_order, rcond=1e-12)
    DEA_TF_T = np.poly1d(z_0)    # Transpose of DEA TF
    id_strain = np.linspace(strain[0], strain[-1], pt)
    corr_V = DEA_TF_T(id_strain)    # Voltage
    for order in range(order_min,order_sup):
        z = np.polyfit(V, corr_V, order, rcond=1e-12)
        TF_np = np.poly1d(z)
        comp_V = torch.from_numpy(TF_np(V))
        comp_strain = DI.strain(comp_V)
        #_, comp_strain = DI.get_stress_strain(comp_V, true_strain=true_strain)
        plt.plot(V*1e-3, comp_strain, label="Compensated, order={}".format(order), linewidth=.7)
    label = '$\mu$={} (MPa)'.format(C.mu*1e-6)
    polyV = DEA_TF_T(strain)    # Voltage
    plt.plot(V*1e-3, strain, 'b', label=label, linewidth=.7)
    plt.plot(polyV*1e-3, strain, 'r', label="Approximated", linewidth=.7)
    plt.plot(V*1e-3, id_strain, 'k--', label="Id", linewidth=.7)
    #plt.plot(V, corr_V, label="Corr V")
    
    plt.grid(True)
    strain_label = 'True strain $\\varepsilon$' if true_strain else 'Engineering Strain$\lambda$'
    plt.xlabel('Voltage $\mathbf{V}$ (kV)')
    #plt.ylabel(strain_label+", normalized by $\mathbf{V}$")
    plt.ylabel(strain_label)
    #plt.ylim(0,0.05)
    plt.title('Target, uniaxial stretch, incompressible')
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig("./test_model_comp.png".format(order), dpi=500, bbox_inches='tight')


    #============================== 
    # Voltage plot
    #============================== 
    #plt.figure(figsize=(5,5))
    #_, strain = DI.get_stress_strain(Vt,true_strain=true_strain)
    #strain = strain.numpy()
    ##strain -= np.min(strain)
    #norm_strain = (V2 / np.max(strain))*strain
    #id_strain = np.linspace(norm_strain[0], norm_strain[-1], pt)
    #plt.plot(V*1e-3, norm_strain*1e-3, label="Stress-strain")
    #plt.plot(V*1e-3, id_strain*1e-3, 'k--', label="Id")
    #plt.plot(V*1e-3, comp_V*1e-3, label="Voltage Mapping")

    #plt.grid(True)
    #plt.xlabel('Voltage $\mathbf{V}$ (kV)')
    ##plt.ylabel(strain_label+", normalized by $\mathbf{V}$")
    #plt.ylabel('Voltage')
    ##plt.ylim(0,0.05)
    #plt.title('Target, uniaxial stretch, incompressible')
    #plt.legend(bbox_to_anchor=(1,1))
    #plt.savefig("./test_model_comp_volt.png".format(order), dpi=500, bbox_inches='tight')








