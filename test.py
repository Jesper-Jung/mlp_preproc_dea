import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import constants as C
import librosa
import librosa.display
from waveshaper import MLP
from actuator import ElastomerActuator
import torch
import torch.nn as nn
from models import Dielectric, NeoHookean
from utils import plot_spectrum, calc_loss, get_gt_str, get_diff

def evaluate(H=None, epoch=0, mode='Test', checkpoint=None):
    #------------------------------  
    # Setup
    #------------------------------  
    if H is None:
        H = MLP(md=C.model, gpu=C.gpu)
        if C.gpu:
            H.cuda()
        if checkpoint:
            H.load_state_dict(checkpoint["model"])
            epoch = checkpoint["epoch"]
            LOSS = checkpoint["loss"]
            ckpt_dir = checkpoint["cwd"]
            print(">  Loaded checkpoint from: {}:".format(ckpt_dir))

    if C.model=='dielectric':
        elastic = Dielectric()
    elif C.model=='neohookean':
        elastic = NeoHookean()

    criterion = {}
    criterion['l1'] = torch.nn.L1Loss()
    criterion['diff'] = torch.nn.MSELoss()

    # voltages to probe
    steps = C.train_samples
    probe_V = np.linspace(C.V_inf, C.V_sup, steps)

    # add gaussian noise
    sigma = ((C.V_sup - C.V_inf) / steps) / 10
    probe_V += np.random.normal(loc=0.0, scale=sigma, size=steps)
    with torch.no_grad():
        #============================== 
        # Test
        #============================== 
        print("="*30)
        print("  {}".format(mode))
        print("="*30)

        # Definition
        prb_V = torch.Tensor(probe_V).unsqueeze(0)
        if C.gpu:
            prb_V = prb_V.cuda()

        # Waveshaping
        gen_V = H(prb_V.repeat(2,1))[0]
        sqrt_V = C.det_waveshaper(prb_V)[0]

        # Calculate strain
        gt_str  = get_gt_str(elastic, prb_V)
        gen_str = elastic.strain(gen_V)
        ori_str = elastic.strain(prb_V)
        sqrt_str = elastic.strain(sqrt_V)

        #============================== 
        # Loss
        #============================== 
        gen_str_ = gen_str.unsqueeze(0).unsqueeze(-1)
        gen_diff = get_diff(gen_str_,prb_V) * C.penalize(epoch)
        gt_diff  = get_diff(gt_str,prb_V) * C.penalize(epoch)

        _gen_str  = gen_str
        _gt_str   = gt_str
        _gen_diff = gen_diff
        _gt_diff  = gt_diff
        vloss_input = { \
            'l1': [_gen_str, _gt_str], \
            'diff': [_gen_diff, _gt_diff], \
        }
        to_cal = {scope: vloss_input[scope] for scope in C.loss_scope}
        vloss_dict = calc_loss(criterion, to_cal)
        loss = sum([vloss_dict[l] for l in C.loss])
        print("..  epoch : {}".format(epoch))
        print("..  loss  : {}".format(loss))

    gt = gt_str.squeeze().cpu().numpy()
    gen = gen_str.squeeze().cpu().numpy()
    ori = ori_str.squeeze().cpu().numpy()
    sqrt = sqrt_str.squeeze().cpu().numpy()
    gen_V = gen_V.squeeze().cpu().numpy()

    str_save_path = os.path.join(C.test_plot,"{}_strain_{}.png".format(mode,epoch))
    vol_save_path = os.path.join(C.test_plot,"{}_voltage_{}.png".format(mode,epoch))

    plt.figure(figsize=(5,5))
    plt.plot(probe_V, gt, '-', label='GT', linewidth=.8)
    plt.plot(probe_V, gen, '-', label='Compensated', linewidth=.8)
    plt.plot(probe_V, ori, '-', label='Original', linewidth=.8)
    plt.plot(probe_V, sqrt, '-', label='sqrt-Compensated', linewidth=.8)
    plt.title('Epoch {}'.format(epoch))
    plt.xlabel('Voltage (V)')
    plt.ylabel('True Strain')
    plt.ylim(gt[0]+1e-4,gt[-1]-1e-4)
    plt.legend()
    plt.savefig(str_save_path, dpi=C.dpi)
    print(">  Saved plot {}".format(str_save_path))

    plt.figure(figsize=(5,5))
    plt.plot(probe_V, probe_V, '-', label='Identity', linewidth=.8)
    plt.plot(probe_V, gen_V, '-', label='Mapped V', linewidth=.8)
    #plt.plot(probe_V, ori, '.-', label='Original')
    plt.title('Epoch {}'.format(epoch))
    plt.xlabel('Voltage (V)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.savefig(vol_save_path, dpi=C.dpi)
    print(">  Saved plot {}".format(vol_save_path))

    return loss


def compensate(checkpoint, H=None, epoch=0):
    dev = torch.device('cuda:0') if C.gpu else torch.device('cpu:0')
    #------------------------------  
    # Setup
    #------------------------------  
    nfft = C.nfft
    sr = C.sr
    dur = C.duration
    sp_len = int(dur*sr)
    xinf = 0
    xsup = min(sr // 2, 2e4)
    yinf = -10
    ysup = 160
    fi = np.arange(0, sr, sr/nfft) # freq. resolution = fs/nfft
    #++++++++++++++++++++++++++++++ 
    # TODO: this should be re-formulated,
    #     - so we actually FT the pressure, not the strain.
    #DEA = ElastomerActuator(C.a_0, C.A, C.r, sr, model=C.model)
    #++++++++++++++++++++++++++++++ 
    if C.model=='dielectric':
        elastic = Dielectric()
    elif C.model=='neohookean':
        elastic = NeoHookean()
    else:
        raise NotImplementedError
    #++++++++++++++++++++++++++++++ 


    if H is None:
        H = MLP(md=C.model, gpu=C.gpu)
        H.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        ckpt_dir = checkpoint["cwd"]
        config = checkpoint["config"]
        print(">  Loaded checkpoint from: {}:".format(ckpt_dir))
        print(config)
        print("."*30)
    else:
        print(">  Using prefix compensator")
        print("..     epoch", epoch)

    with torch.no_grad():
        print("="*30)
        print("  Compensate")
        print("="*30)
        print("..  Epoch {}.".format(epoch))
        #------------------------------ 
        # Definition
        #------------------------------ 
        freqs  = torch.Tensor(C.Test_FREQ).unsqueeze(1)/sr
        freqs  = freqs.cuda().repeat(1,sp_len) if C.gpu else freqs.repeat(1,sp_len)
        omega  = torch.cumsum(2*np.pi*freqs, dim=1)
        d_omg  = 2*np.pi*freqs*sr
        sin = C.Vdc + C.Vpp * torch.sin(omega)
        cos = C.Vdc + C.Vpp * torch.cos(omega)

        signal = sin

        #============================== 
        # Waveshaping
        #============================== 
        gen = H(signal)

        #------------------------------ 
        # Pressure amplitude
        #------------------------------ 
        print("..  Applying voltage: DC {} V, AC {} V".format(C.Vdc, C.Vpp))
        #++++++++++++++++++++++++++++++ 
        # TODO: this should be re-formulated,
        #     - so we actually FT the pressure, not the strain.
        #DEA = ElastomerActuator(C.a_0, C.A, C.r)
        #prs = DEA.pressure(gen, d_omg)
        #ori = DEA.pressure(signal, d_omg)
        #gt = torch.cos(omega)
        #++++++++++++++++++++++++++++++ 
        prs = elastic.strain(gen)
        ori = elastic.strain(signal)
        gt  = get_gt_str(elastic, signal)
        prs_dfm = (1 + prs) * C.z_0
        ori_dfm = (1 + ori) * C.z_0
        gt_dfm  = (1 +  gt) * C.z_0
        #++++++++++++++++++++++++++++++ 

        OUT = prs.cpu().numpy() * 1e-4
        ORI = ori.cpu().numpy() * 1e-4
        GTS = gt.cpu().numpy()  * 1e-4
        GEN = gen.cpu().numpy()

        GTV = GTS*C.Vpp + C.Vdc
        #GTS = GTS*1e-9 if C.use_de else GTS*1e-4

        #============================== 
        # Loss
        #============================== 
        valid_loss = nn.MSELoss()
        loss = valid_loss(prs, gt)
        print("..  loss  : {}".format(loss.item()))

    for i,freq in enumerate(C.Test_FREQ):
        print("..  Writing "+str(freq)+" Hz")
        #sf.write(os.path.join(C.save_plot(epoch),"w_"+str(freq)+"_Hz.wav"), OUT[i], C.sr, subtype='PCM_24')

        #------------------------------ 
        # Plot Waveform
        #------------------------------ 
        periods = 5
        prd_smpl = periods*C.sr / freq
        plt.figure(figsize=(12,6))
        plt.subplot(211)
        plt.title("{} Hz ({} periods plotted, epoch {})".format(str(freq), periods, str(epoch)))
        plt.plot(GTV[i], 'r--', label="Original Input V", linewidth=.7)
        plt.plot(GEN[i], 'k--', label="Mapped V", linewidth=.7)
        plt.xlim(0,prd_smpl)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.subplot(212)
        plt.plot(ORI[i], label="Raw Output Waveshape", color="r", linewidth=.7)
        plt.plot(OUT[i], label="Compensated Waveshape", color="k", linewidth=.7)
        plt.xlim(0,prd_smpl)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(C.save_plot(epoch),"wf_DEA_"+str(freq)+"_Hz.png"), dpi=C.dpi, bbox_inches='tight')
        plt.close()

        #------------------------------ 
        # Plot Spectrum
        #------------------------------ 
        plt.figure(figsize=(12,3))
        plot_spectrum(fi,ORI[i],nfft,label="Original", color="r", linewidth=.7)
        plot_spectrum(fi,GTS[i],nfft,label="Desired", color="b", linewidth=.7)
        plot_spectrum(fi,OUT[i],nfft,label="Compensated", color="k", linewidth=.7)
        plt.title("Spectrum - {} Hz (epoch {})".format(str(freq), str(epoch)))
        plt.xlabel('frequency (Hz)')
        plt.ylabel('dB SPL')
        if C.log_xscale:
            plt.xscale('log')
        plt.xlim(xinf,xsup)
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(C.save_plot(epoch),"spec_DEA_"+str(freq)+"_Hz.png"), dpi=C.dpi, bbox_inches='tight')
        plt.close()


    print("Done.")
    print("="*30)
 

if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings("ignore")

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    dev = torch.device('cuda:0') if C.gpu else torch.device('cpu:0')
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load")
    parser.add_argument("--evaluate", type=str2bool, default='false', help="Whether to evaluate trained NN")
    parser.add_argument("--compensate", type=str2bool, default='false', help="Whether to compensate using trained NN")
    args = parser.parse_args()
    if args.compensate:
        assert not (args.checkpoint is None), "Checkpoint to load weights for compensation should be given"

    # LaTeX Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Test
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        if args.evaluate:
            evaluate(checkpoint=ckpt)
        if args.compensate:
            compensate(checkpoint=ckpt)
    else:
        evaluate()
    

