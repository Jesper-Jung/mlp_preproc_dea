import os
import librosa
import librosa.display
import constants as C
import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
from models import Dielectric, NeoHookean
import scipy

def renormalize_np(x, ac, dc):
    x = x / np.max(np.absolute(x))
    x = dc + ac * x
    return x

def plot_spectrum(freq, sig, nfft, label=None, color=None, linewidth=None):
    spec = np.fft.fft(sig / C.P_0, nfft)
    dBspec = 20*np.log10(np.absolute(spec) + np.finfo(float).eps)
    plt.plot(freq, dBspec, label=label, color=color, linewidth=linewidth)

def log_waveform(x):
    return 10*torch.log(x + 1e-12)

def tan_waveform(x, amp, eps=1e-2):
    asymp = (np.pi/2) / (amp+eps)
    return 10*torch.tan(asymp*x)

def calc_loss(criterion, to_cal):
    loss_dict = {}
    for key, value in to_cal.items():
        loss_dict[key] = criterion[key](*value)
    return loss_dict

def elastic_deformation(x):
    V = C.Vdc + C.Vpp * x
    strss = C.e_0 * C.e_r * ((V / C.z_0)**2)
    lam = 1 + (-1/C.Y) * strss
    return (C.z_0/2) * lam
            
def cplx_mult(x,y, dev='cuda:0'):
    out = torch.empty(x.shape, device=dev)
    out[:,...,0] = x[:,...,0] * y[:,...,0] - x[:,...,1] * y[:,...,1]
    out[:,...,1] = x[:,...,0] * y[:,...,1] - x[:,...,1] * y[:,...,0]
    return out

def get_gt_str(md,signal):
    str_1 = md.strain(torch.Tensor([C.V_inf])).squeeze()
    str_2 = md.strain(torch.Tensor([C.V_sup])).squeeze()
    r = (str_2 - str_1)/(C.V_sup - C.V_inf)
    return r * (signal - C.V_inf) + str_1

def get_diff(x,t,axis=-1):
    dx = torch.roll(x[axis],-1,0) - x[axis]
    dt = torch.roll(t[axis],-1,0) - t[axis]
    dx[-1] = dx[-2]
    dt[-1] = dt[-2]
    return dx / dt

#============================== 
# Preprocess Data
#============================== 

def save_prep(prep, sf_prep, orig=None, sf_orig=None, sr=C.sr):
    print(".. Writing",sf_prep)
    _max = np.max(prep)
    _min = np.min(prep)
    _vpp = _max - _min
    _vdc = _max - (_vpp / 2)
    
    #------------------------------ 
    # Normalize in amplitude 1.
    prep = (prep - _vdc) / ((_vpp+1e-3) / 2)    # [-1, 1]
    #------------------------------ 
    # Normalize in DAQ amplify rate
    #assert _max*1e-3 < C.DAQ_out_amp, "Signal exceeds DAQ output amplitude"
    #prep = (prep - _vdc) / (_vpp / 2)    # [-1, 1]
    #prep = (prep*(_vpp*0.5e-3) + _vdc*1e-3) / C.DAQ_out_amp
    #------------------------------ 
    print(".. Preprocessed signal: {} +- {} (kV)".format(round(_vdc*1e-3,3), round(_vpp*1e-3,3)))
    sf.write(sf_prep, prep, sr, subtype='PCM_24')
    print(".. Saved:")
    print("\t {}".format(sf_prep))
    if orig is not None:
        #_orig = renormalize_np(orig, _vpp*1e-3, _vdc*1e-3) / C.DAQ_out_amp
        _max = np.max(orig)
        _min = np.min(orig)
        _vpp = _max - _min
        _vdc = _max - (_vpp / 2)
        _orig = (orig - _vdc) / (_vpp / 2)    # [-1, 1]
        sf.write(sf_orig, _orig, sr, subtype='PCM_24')
        print("\t {}".format(sf_orig))
    print("-"*30)

def prep_song(H, srpath, simulate):
    print("="*30)
    print(">  Preprocess Loaded Data..")
    print("="*30)
    for tar_sr in srpath:
        dpath = glob(os.path.join(tar_sr,"*.wav"))
        for dfile in dpath:
            fname = dfile.split('/')[-1]
            sname = fname.split(".wav")[0]+"_prep.wav"
            sr = int(dfile.split("sr_")[-1].split('/')[0])
            sr_spath = os.path.join(spath,"sr_"+str(sr))
            os.makedirs(sr_spath, exist_ok=True)

            sfile_prep = os.path.join(sr_spath,sname)
            sfile_orig = os.path.join(sr_spath,fname)
            
            print(">  Loaded {}".format(dfile))
            o_signal, _ = librosa.load(dfile, sr=sr)
            if resample == True:
                o_signal = librosa.resample(o_signal, orig_sr = sr, target_sr = C.sr)
                sr = C.sr
            signal = renormalize_np(o_signal, C.Vpp, C.Vdc)
            signal = torch.Tensor(signal)
            signal = signal.unsqueeze(0)
            print(".. sr = {}".format(sr))
            if C.gpu:
                signal = signal.cuda()

            preproc = H(signal)

            preproc = preproc.squeeze()
            signal = signal.squeeze()
            preproc = preproc.cpu().numpy()

            save_prep(preproc,sfile_prep,o_signal,sfile_orig, sr=sr)

def prep_sine(H, spath, simulate):
    print("="*30)
    print(">  Preprocess Pure Sine..")
    print("="*30)
    sr = C.sr
    sr_spath = os.path.join(spath,"sr_"+str(sr))
    os.makedirs(sr_spath, exist_ok=True)
    dur = C.duration
    sp_len = int(dur*sr)
    freqs  = torch.Tensor(C.Test_FREQ).unsqueeze(1)/sr
    freqs  = freqs.cuda().repeat(1,sp_len) if C.gpu else freqs.repeat(1,sp_len)
    omega  = torch.cumsum(2*np.pi*freqs, dim=1)
    d_omg  = 2*np.pi*freqs*sr
    sin = C.Vdc + C.Vpp * torch.sin(omega)
    if C.use_cplx:
        cos = C.Vdc + C.Vpp * torch.cos(omega)
        sin = sin.unsqueeze(-1)
        cos = cos.unsqueeze(-1)
        signal = torch.cat((cos,sin),-1)
    else:
        signal = sin
    o_signal = signal.cpu().numpy()

    preproc = H(signal)
    preproc = preproc.squeeze().cpu().numpy()

    for i,freq in enumerate(C.Test_FREQ):
        p_sname = str(freq)+"_prep.wav"
        o_sname = str(freq)+"_orig.wav"
        sfile_prep = os.path.join(sr_spath,p_sname)
        sfile_orig = os.path.join(sr_spath,o_sname)

        save_prep(preproc[i],sfile_prep, o_signal[i],sfile_orig)


def prep_chirp(H, spath, simulate, filter=None):
    print("="*30)
    print(">  Preprocess Chirp..")
    print("="*30)
    sr = C.sr
    sr_spath = os.path.join(spath,"sr_"+str(sr))
    os.makedirs(sr_spath, exist_ok=True)
    dur = C.infer_dur
    sp_len = int(dur*sr)
    freqs  = torch.linspace(C.CH_INF,C.CH_SUP,sp_len)/sr
    freqs  = freqs.unsqueeze(0).cuda() if C.gpu else freqs.unsqueeze(0)
    omega  = torch.cumsum(2*np.pi*freqs, dim=1)
    d_omg  = 2*np.pi*freqs*sr
    sin = C.Vdc + C.Vpp * torch.sin(omega)
    if C.use_cplx:
        cos = C.Vdc + C.Vpp * torch.cos(omega)
        sin = sin.unsqueeze(-1)
        cos = cos.unsqueeze(-1)
        signal = torch.cat((cos,sin),-1)
    else:
        signal = sin
    o_signal = signal.squeeze().cpu().numpy()

    bz = int(dur*10)
    signal = torch.reshape(signal,(bz, sr//int(bz / dur)))
    preproc = torch.zeros_like(signal)
    for ii in range(bz):
        if filter:
            signal = signal.detach().cpu().numpy()
            ir = np.load(filter)
            ll = signal[ii].shape[0]
            signal[ii] = scipy.signal.convolve(signal[ii],ir)[:ll]
            signal = torch.tensor(signal).cuda()
        preproc[ii] = H(signal[ii])

    if simulate:
        #++++++++++++++++++++++++++++++ 
        # TODO: this should be re-formulated,
        #     - so we actually FT the pressure, not the strain.
        #DEA = ElastomerActuator(C.a_0, C.A, C.r, sr, model=C.model)
        #prs = DEA.pressure(gen, d_omg)
        #ori = DEA.pressure(signal, d_omg)
        #gt = torch.cos(omega)
        #++++++++++++++++++++++++++++++ 
        if C.model=='dielectric':
            elastic = Dielectric()
        else:
            raise NotImplementedError
        dea_out = elastic.strain(preproc)
        dea_out = torch.reshape(dea_out,(1, int(dur*sr)))
        dea_out = dea_out.squeeze().cpu().numpy()
        _max = np.max(dea_out)
        _min = np.min(dea_out)
        _vpp = _max - _min
        _vdc = _max - (_vpp / 2)
        dea_out_nrml = (dea_out - _vdc) / ((_vpp+1e-4) / 2)    # [-1, 1]
        sfile_dout = os.path.join(sr_spath,"chirp_DEAout.wav")
        sf.write(sfile_dout, dea_out_nrml, C.sr, subtype='PCM_24')
        print(".. Saved {}".format(sfile_dout))

    preproc = torch.reshape(preproc,(1, int(dur*sr)))
    preproc = preproc.squeeze().cpu().numpy()
    sfile_prep = os.path.join(sr_spath,"chirp_prep.wav")
    sfile_orig = os.path.join(sr_spath,"chirp_orig.wav")

    save_prep(preproc,sfile_prep,o_signal,sfile_orig)
    
    
def prep_sqrt(model, spath, nfft=2**10):
    print("="*30)
    print(">  Preprocess Chirp..(sqrt)")
    print("="*30)
    sr = C.sr
    sr_spath = os.path.join(spath,"sr_"+str(sr))
    os.makedirs(sr_spath, exist_ok=True)
    dur = C.infer_dur
    sp_len = int(dur*sr)
    freqs  = torch.linspace(C.CH_INF,C.CH_SUP,sp_len)/sr
    freqs  = freqs.unsqueeze(0).cuda() if C.gpu else freqs.unsqueeze(0)
    omega  = torch.cumsum(2*np.pi*freqs, dim=1)
    d_omg  = 2*np.pi*freqs*sr
    sin = C.Vdc + C.Vpp * torch.sin(omega)

    preproc = C.comp_filter(sin)
    x = C.comp_filter(sin)
    prep_nd = preproc.squeeze().cpu().numpy()
    sfile_prep = os.path.join(sr_spath,"chirp_prep_sqrt.wav")

    save_prep(prep_nd,sfile_prep)

    #++++++++++++++++++++++++++++++ 
    # TODO: this should be re-formulated
    #     - so we actually FT the pressure, not the strain.
    #prs = model.pressure(preproc, d_omg) * 1e5
    #OUT = prs.squeeze().cpu().numpy()
    #++++++++++++++++++++++++++++++ 
    prs = model.strain(preproc)
    OUT = prs.squeeze().cpu().numpy()

    spec, phase = librosa.magphase(librosa.stft(OUT, n_fft=nfft, win_length=nfft))
    plt.figure(figsize=(9,5)) 
    librosa.display.specshow(np.log(spec + 0.001), cmap='viridis', x_axis='time', y_axis='linear', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title("Square-root")
    plt.ylim(0,20000)
    plt.savefig(sfile_prep[:-4])

if __name__ == '__main__':
    from actuator import ElastomerActuator
    from models import Dielectric, NeoHookean
    #++++++++++++++++++++++++++++++ 
    # TODO: this should be re-formulated
    #     - so we actually FT the pressure, not the strain.
    #model = ElastomerActuator(C.a_0, C.A, C.r, C.sr, model=C.model)
    #++++++++++++++++++++++++++++++ 
    if C.model=='dielectric':
        model = Dielectric()
    elif C.model=='neohookean':
        model = NeoHookean()
    spath = os.path.join("./Data/Save")
    prep_sqrt(model, spath, nfft=2**14)





