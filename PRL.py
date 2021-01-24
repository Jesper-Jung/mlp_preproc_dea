import numpy as np
import torch
from torch.autograd import Variable

class power_function_(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, power):        
        ctx.save_for_backward(input, power) # save to reuse in backward
        p = power.expand_as(input)
        return input**p

    @staticmethod
    def backward(ctx, grad_output):   # grad_output comes from backward
        x, p = ctx.saved_variables # from forward
        B = grad_output.shape[0]    # num of batch
        N = grad_output.shape[1]    # input signal length
        l = grad_output.shape[2]    # layer dim

        # dx <-- d(x**p) / [p * x**(p-1)]
        grad_input = grad_output / (p * x**(p - 1) + 1e-12)

        # dp <-- d(x**p) / x**p
        update_p = grad_output / (x**p + 1e-12)
        grad_power_ = torch.sum(update_p, dim=1) / N
        grad_power  = torch.sum(grad_power_, dim=0) / B

        return grad_input, grad_power

class pRootLayer(torch.nn.Module): 
    def __init__(self, p=.5, alpha=1e-6):
        super(pRootLayer, self).__init__()
        self.front = torch.nn.CELU(alpha)
        self.alpha = alpha*2

        self.power = torch.nn.Parameter(torch.Tensor(1))
        self.power.data.fill_(p)

    def forward(self, x):
        x = self.front(x) + self.alpha
        return power_function_.apply(x, self.power)



