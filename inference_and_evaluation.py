import os, sys
import numpy as np
import h5py
from tqdm import tqdm

import torch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import rfcutils
from omegaconf import OmegaConf
from src.config_torchwavenet import Config, parse_configs
from src.torchwavenet import Wave
from src.soi_autoencoder import WaveQPSK, WaveOFDM

get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960
n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len//16)
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len//16)
        demod_soi = rfcutils.qam16_matched_filter_demod
    elif soi_sig_type ==  'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len//4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len//80)
        _,_,_,RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len//80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi, demod_soi

def load_model(id_string, model_file, soi_type):
    if id_string.lower() == 'wavenet':
        cfg = OmegaConf.load("src/configs/wavenet.yml")
        cfg: Config = Config(**parse_configs(cfg, None))
        nn_model = Wave(cfg.model).cuda()
        nn_model.load_state_dict(torch.load(model_file)['model'])
        return nn_model
    elif id_string.lower().startswith('wave-sae'):
        yml = id_string.lower()
        if yml.endswith("_ft"): yml = yml[:-3]
        if yml[-1] in [str(i) for i in range(10)]: yml = yml[:-1]
        cfg = OmegaConf.load(f"src/configs/{yml}.yml")
        cfg: Config = Config(**parse_configs(cfg, None))
        if soi_type == 'QPSK':
            nn_model = WaveQPSK(cfg.model, cfg.sae).cuda()
        elif soi_type =='OFDMQPSK':
            nn_model = WaveOFDM(cfg.model, cfg.sae).cuda()
        nn_model.load_state_dict(torch.load(model_file)['model'])
        return nn_model
    elif id_string.lower() == 'none':
        return None
    else:
        raise ValueError(f'Unknown model identifier {id_string}')

def run_inference(all_sig_mixture, soi_type, nn_model):
    # inference pipeline
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)

    if nn_model is not None:
        with torch.no_grad():
            nn_model.eval()
            all_sig1_out = []
            all_bit1_out = []
            bsz = 10
            for i in tqdm(range(all_sig_mixture.shape[0]//bsz), leave=False):
                sig_input = torch.from_numpy(all_sig_mixture[i*bsz:(i+1)*bsz])
                sig_input = torch.view_as_real(sig_input).transpose(-2,-1).to('cuda')
                sig1_out = nn_model(sig_input)
                if isinstance(sig1_out, tuple):
                    all_bit1_out.append(sig1_out[1].detach().cpu().numpy())
                    sig1_out = sig1_out[0]
                all_sig1_out.append(sig1_out.transpose(1,2).detach().cpu().numpy())
    else: # direct demodulation of all_sig_mixture
        all_sig1_out = np.stack([all_sig_mixture.real, all_sig_mixture.imag], axis=-1)
        all_bit1_out = []
    sig1_out = tf.concat(all_sig1_out, axis=0)
    sig1_est = tf.complex(sig1_out[:,:,0], sig1_out[:,:,1])
    if len(all_bit1_out) > 0:
        thr = 0.0 if nn_model.sigmoid else 0.5
        bit1_out = np.concatenate(all_bit1_out, axis=0)
        bit1_out = (bit1_out > thr).astype(np.int32)
    else:
        bit1_out = None

    bit_est = []
    for idx, sinr_db in tqdm(enumerate(all_sinr), leave=False):
        bit_est_batch, _ = demod_soi(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_est.append(bit_est_batch)
    bit_est = tf.concat(bit_est, axis=0)
    sig1_est, bit_est = sig1_est.numpy(), bit_est.numpy()
    return sig1_est, bit_est, bit1_out

def run_demod_test(sig1_est, bit1_est, all_sig1, all_bits1):    
    # demod pipeline
    def eval_mse(all_sig_est, all_sig_soi):
        assert all_sig_est.shape == all_sig_soi.shape, 'Invalid SOI estimate shape'
        return np.mean(np.abs(all_sig_est - all_sig_soi)**2, axis=1)
    
    def eval_ber(bit_est, bit_true):
        ber = np.sum((bit_est != bit_true).astype(np.float32), axis=1) / bit_true.shape[1]
        assert bit_est.shape == bit_true.shape, 'Invalid bit estimate shape'
        return ber

    all_mse, all_ber = [], [] 
    for idx, sinr in enumerate(all_sinr):
        batch_mse =  eval_mse(sig1_est[idx*n_per_batch:(idx+1)*n_per_batch], all_sig1[idx*n_per_batch:(idx+1)*n_per_batch])
        bit_true_batch = all_bits1[idx*n_per_batch:(idx+1)*n_per_batch]
        batch_ber = eval_ber(bit1_est[idx*n_per_batch:(idx+1)*n_per_batch], bit_true_batch)
        all_mse.append(batch_mse)
        all_ber.append(batch_ber)

    all_mse, all_ber = np.array(all_mse), np.array(all_ber)
    mse_mean = 10*np.log10(np.mean(all_mse, axis=-1))
    ber_mean = np.mean(all_ber, axis=-1)
    return mse_mean, ber_mean

def main(soi_type, interference_sig_type, id_string, testset_identifier):
    # load evaluation data
    with h5py.File(os.path.join('dataset', f'{testset_identifier}_Dataset_{soi_type}_{interference_sig_type}.h5'), 'r') as hf:
        all_sig_mixture = np.array(hf.get('mixtures'))
        all_sig1 = np.array(hf.get('soi'))
        all_bits1 = np.array(hf.get('bits'))

    # load model
    nn_model = load_model(id_string, os.path.join('models', f'{soi_type}_{interference_sig_type}_{id_string}', 'weights.pt'), soi_type)

    # inference on evaluation data
    mse_ber = []
    sig1_est, bit1_est, bit1_out = run_inference(all_sig_mixture, soi_type, nn_model)
    mse_mean, ber_mean = run_demod_test(sig1_est, bit1_est, all_sig1, all_bits1)
    if bit1_out is not None:
        _, ber_mean_out = run_demod_test(sig1_est, bit1_out, all_sig1, all_bits1)
        print('BER - demod:', np.mean(10*np.log10(ber_mean+1e-6)), 'output:', np.mean(10*np.log10(ber_mean_out+1e-6)))
        ber_mean = ber_mean_out

    # save results
    mse_ber = np.stack([mse_mean, ber_mean], axis=0)
    np.save(os.path.join('outputs', f'{id_string}_{testset_identifier}_{soi_type}_{interference_sig_type}_results'), mse_ber)

if __name__ == "__main__":
    # input
    id_string = [sys.argv[1]] if len(sys.argv) > 1 else ['none', 'wavenet', 'wave-sae', 'wave-sae_ft']
    soi_type = [sys.argv[2]] if len(sys.argv) > 2 else ['QPSK', 'OFDMQPSK']
    interference_sig_type = [sys.argv[3]] if len(sys.argv) > 3 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    testset_identifier = sys.argv[4] if len(sys.argv) > 4 else 'TestSet1Example'

    # call main
    for id in id_string:
        for s in soi_type:
            for i in interference_sig_type:
                try:
                    main(s, i, id, testset_identifier)
                except FileNotFoundError as fnfe:
                    print(fnfe)
                    continue