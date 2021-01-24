import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import constants as C
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import scipy.signal
from waveshaper import MLP
from models import Dielectric, NeoHookean, Gent
from test import evaluate, compensate
from utils import calc_loss, get_gt_str, get_diff

def save_checkpoint(model, optimizer, scheduler, \
                    epoch, LOSS, TOT_LOSS, VAL_LOSS, POW,\
                    ckpt_dir, config, name='filter'):
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss": LOSS,
        "tot_loss": TOT_LOSS,
        "val_loss": VAL_LOSS,
        "p": POW,
        "cwd": ckpt_dir,
        "config": config,
    }
    checkpoint_path = os.path.join(ckpt_dir,'{}.pt'.format(name))
    torch.save(checkpoint_state, checkpoint_path)
    print("**  Saved checkpoint: {}".format(checkpoint_path))


def train(lr, dev, config, checkpoint=None):
    print("="*30)
    print("  Train")
    print("="*30)
    H = MLP(md=C.model, gpu=C.gpu)
    if C.gpu:
        H.cuda()

    criterion = {}
    criterion['l1'] = torch.nn.L1Loss()
    criterion['diff'] = torch.nn.MSELoss()

    optimizer = optim.Adam(H.parameters(), lr=lr)
    scheduler = StepLR(optimizer,
                       step_size=C.lr_decay[0],
                       gamma=C.lr_decay[1])

    sp_len = C.train_samples
    signal = torch.empty((C.batch_size,sp_len),device=dev)

    if C.model=='dielectric':
        elastic = Dielectric()
    elif C.model=='neohookean':
        elastic = NeoHookean()
    elif C.model=='gent':
        elastic = Gent()

    POW = []
    LOSS = []
    TOT_LOSS = []
    VAL_LOSS = []
    i_e = 1     # initial epoch
    if checkpoint:
        H.load_state_dict(checkpoint["model"])
        i_e = checkpoint["epoch"] + 1
        LOSS = checkpoint["loss"]
        TOT_LOSS = checkpoint["tot_loss"]
        VAL_LOSS = checkpoint["val_loss"]
        POW = checkpoint["p"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        ckpt_dir = checkpoint["cwd"]
        print(">  Loaded checkpoint from: {}:".format(ckpt_dir))


    #============================== 
    # Train
    #============================== 
    EPOCH = np.arange(0, C.epochs+1, 1)
    for epoch in EPOCH[i_e:]:  # loop over the dataset multiple times
        if C.cout_epoch > 0 and epoch % C.cout_epoch == 0:
            print("-"*30)
            print(">   Epoch : {}".format(epoch))
        #------------------------------ 
        # Definition
        #------------------------------ 
        random = torch.rand(C.batch_size,sp_len)             # Random data
        linspc = torch.linspace(C.V_inf,C.V_sup,sp_len)      # linspace
        rseed = torch.rand(sp_len)*(.9*(C.V_sup - C.V_inf)/sp_len)
        rseed[-1] = 0
        linspc = linspc + rseed
        linspc = torch.unsqueeze(linspc,0)
        random = (C.V_sup - C.V_inf) * random + C.V_inf
        signal = torch.cat((random,linspc),0)
        if C.gpu:
            signal = signal.cuda()

        #------------------------------ 
        # Compensate
        #------------------------------ 
        gen = H(signal)

        #------------------------------ 
        # Calculate strain
        #------------------------------ 
        gen_str = elastic.strain(gen)
        gt_str  = get_gt_str(elastic, signal)

        #------------------------------ 
        # Calculate Differential
        #------------------------------ 
        gen_diff = get_diff(gen_str,signal) * C.penalize(epoch)
        gt_diff  = get_diff(gt_str,signal) * C.penalize(epoch)

        _gen_str  = gen_str
        _gt_str   = gt_str 
        _gen_diff = gen_diff
        _gt_diff  = gt_diff
        loss_input = { \
            'l1': [_gen_str, _gt_str], \
            'diff': [_gen_diff, _gt_diff], \
        }

        to_cal = {scope: loss_input[scope] for scope in C.loss_scope}
        loss_dict = calc_loss(criterion, to_cal)
        loss = sum([loss_dict[l] for l in C.loss])
        tot_loss = [loss_dict[l] for l in C.loss_scope]

        loss.mean().backward()
        optimizer.step()
        scheduler.step()

        if epoch % C.loss_epoch == 0:
            LOSS.append(loss.item())
            if C.use_PRL:
                for i in range(len(C.channels)-2):
                    POW.append(H.state_dict()['dnn.{}.power'.format(2*i+1)].item())
            TOT_LOSS.append(tot_loss)
        if C.cout_epoch > 0 and epoch % C.cout_epoch == 0:
            print("..  lr    : {}".format(scheduler.state_dict()['_last_lr'][0]))
            print("..  loss  : {}".format(LOSS[-1]))

        #============================== 
        # Validate
        #============================== 
        if epoch % C.valid_epoch == 0:
            val_loss = evaluate(H=H, epoch=epoch, mode="Valid")
            VAL_LOSS.append(val_loss)

        if epoch % C.test_epoch == 0:
            compensate(None, H=H, epoch=epoch)

        #============================== 
        # Save Checkpoint
        #============================== 
        if epoch % C.ckpt_epoch == 0:
            ckpt_dir = C.get_ckpt_dir(epoch, C.model, C.material, C.gpu)
            save_checkpoint(H, optimizer, scheduler,\
                epoch, LOSS, TOT_LOSS, VAL_LOSS, POW,\
                ckpt_dir, config, name=C.ckpt_name)
            recent_ckpt = os.path.join(ckpt_dir,'{}.pt'.format(C.ckpt_name))
            rcnt_ckpt_epc = epoch

        #============================== 
        # Plot
        #============================== 
        if epoch % C.plot_epoch == 0:
            plt.figure()
            le = np.arange(0,epoch,C.loss_epoch) + C.loss_epoch
            ve = np.arange(0,(epoch//C.plot_epoch)*C.plot_epoch,C.plot_epoch) + C.plot_epoch
            for ss, scope in enumerate(C.loss_scope):
                if scope == 'l1':
                    sc = '$L_1$'
                    col = 'k:'
                elif scope == 'l2':
                    sc = '$L_2$'
                    col = 'k--'
                elif scope == 'diff':
                    sc = '$\Delta\\varepsilon$'
                    col = 'k-.'
                else:
                    sc = scope
                plt.plot(le,np.array(TOT_LOSS).T[ss], col, label=sc)
            plt.plot(le,LOSS, 'k', label='train tot.')
            if epoch % C.valid_epoch:
                for _ in range(C.n_vp - 1):
                    VAL_LOSS.append(np.nan)
            plt.plot(ve,VAL_LOSS, 'ko', fillstyle='none', label='valid')
            plt.title("Batch {}; LR {}, decay {} \% every {} epochs".format(C.batch_size, C.learning_rate, C.lr_decay[1]*100, C.lr_decay[0]))
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(C.plot_path,"loss.png"))
            plt.close()

            if C.use_PRL:
                nprl = len(C.channels)-2
                plt.figure()
                for i in range(nprl):
                    plt.plot(le, POW[i::nprl], label='$p_{}$'.format(i+1))
                plt.xlabel('Epochs')
                plt.ylabel('Power')
                plt.legend()
                plt.savefig(os.path.join(C.plot_path,"p.png"))
                plt.close()

    print("..  done")
    print("="*30)

    return LOSS, VAL_LOSS

    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    dev = torch.device('cuda:0') if C.gpu else torch.device('cpu:0')

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to resume")
    args = parser.parse_args()

    config  = " Material\t: {}\n Total Epochs\t: {}\n".format(C.material, C.epochs)
    config += " LR\t\t: {}\n LR Decay\t: {}\n".format(C.learning_rate, C.lr_decay)
    config += " Batch\t\t: {}\n Layer Width\t: {} \n".format(C.batch_size, C.channels)
    config += " p Init\t\t: {}\n ".format(C.pow_init)

    # log config
    print("."*30)
    print(config)
    print("."*30)

    # LaTeX Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #with torch.autograd.set_detect_anomaly(True):
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        l, vl = train(lr=C.learning_rate, dev=dev, config=config, checkpoint=ckpt)
    else:
        l, vl = train(lr=C.learning_rate, dev=dev, config=config)
   
    min_li  = np.argmin(l)
    vlwonan = np.array(vl)[C.n_vp-1::C.n_vp]
    min_vli = np.argmin(vlwonan)
    
    print(" * Minimum train loss: {} (epoch {})".format(l[min_li], min_li*C.loss_epoch))
    print(" * Minimum valid loss: {} (epoch {})".format(vlwonan[min_vli], min_vli*C.valid_epoch))


