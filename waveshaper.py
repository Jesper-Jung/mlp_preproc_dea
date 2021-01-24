import constants as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF
import scipy.signal
from models import Dielectric, NeoHookean
import numpy as np
from PRL import pRootLayer
from utils import prep_song, prep_sine, prep_chirp

class MLP(nn.Module):
    def __init__(self, md='dielectric', gpu=True):
        super(MLP, self).__init__()

        self._cuda = gpu
        self.dev = torch.device('cuda:0') if self._cuda else torch.device('cpu:0')

        #------------------------------ 
        # Define Network
        #------------------------------ 
        dnn = []
        for c in range(len(C.channels)-2):
            dnn.append(nn.Linear(C.channels[c], C.channels[c+1]))
            if C.use_PRL:
                dnn.append(pRootLayer(p=C.pow_init[c], alpha=1e-12))
            else:
                dnn.append(nn.Sigmoid())
        dnn.append(nn.Linear(C.channels[-2], C.channels[-1]))
        dnn.append(nn.ReLU())

        _dnn = nn.Sequential(*dnn)
        self.dnn = _dnn.cuda() if self._cuda else _dnn

        def init_fc_weights(layer):
            if type(layer) == nn.Linear:
                torch.nn.init.constant_(layer.weight,5.67e-2)
                layer.bias.data.fill_(1e-4)
        self.dnn.apply(init_fc_weights)

    def forward(self, signal):
        """
        Arguments
            signal: Input signal of shape (batch_size, samples)
        Return
            Compensated signal of shape (batch_size, samples)
        """
        signal = (signal - C.V_inf) / (C.V_sup - C.V_inf)
        #print("***", signal)
        signal = signal.unsqueeze(-1)
        out = self.dnn(signal)
        #print("***", out)
        #out = out - torch.min(out)    # (0, ..]
        #out = out / torch.max(out)    # (0,  1]
        out = (C.V_sup - C.V_inf) * out + C.V_inf
        return out.squeeze()



if __name__ == '__main__':

    import os
    from glob import glob
    import argparse
    import torch
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./Data/Load", help="Dir path to load data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--sine", type=str2bool, default='false', help="Preprocess pure sine")
    parser.add_argument("--chirp", type=str2bool, default='false', help="Preprocess chirp")
    parser.add_argument("--song", type=str2bool, default='false', help="Preprocess song")
    parser.add_argument("--simulate", type=str2bool, default='false', help="Also write output of DEA simulator")
    parser.add_argument("--resample", type=str2bool, default='false', help="Resample to constants.sr")
    parser.add_argument("--filter", type=str, default='./ir_90000.npy', help="iir filter for FR compensation")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    do_sine = args.sine
    do_chirp = args.chirp
    do_song = args.song
    simulate = args.simulate
    resample = args.resample
    filter_tap = args.filter
    H = MLP(md=C.model, gpu=C.gpu)
    if C.gpu:
        H.cuda()
    H.load_state_dict(ckpt["model"])
    ckpt_dir = ckpt["cwd"]
    epoch = ckpt["epoch"]
    print(">  Loaded checkpoint from: {}:".format(ckpt_dir))

    spath = os.path.join("./Data/Save", str(epoch))
    srpath = glob(os.path.join(args.data,"*"))
    with torch.no_grad():
        if do_song:
            prep_song(H, srpath,simulate)
        if do_sine:
            prep_sine(H,spath,simulate)
        if do_chirp:
            prep_chirp(H,spath,simulate,filter_tap)
            #prep_chirp(H,spath,simulate,None)

