import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np
import random
import h5py
from tqdm import tqdm

import rfcutils
import tensorflow as tf

get_db = lambda p: 10*np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2, axis=-1)
get_sinr = lambda s, i: get_pow(s)/get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s,i))

sig_len = 40960
n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

seed_number = 0

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


def generate_demod_testmixture(soi_type, interference_sig_type):

    generate_soi, demod_soi = get_soi_generation_fn(soi_type)
    with h5py.File(os.path.join('dataset', 'testset1_frame', interference_sig_type+'_test1_raw_data.h5'),'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")

    random.seed(seed_number)
    np.random.seed(seed_number)
    tf.random.set_seed(seed_number)

    all_sig_mixture, all_sig1, all_bits1, meta_data = [], [], [], []
    for idx, sinr in tqdm(enumerate(all_sinr)):
        sig1, _, bits1, _ = generate_soi(n_per_batch, sig_len)
        sig2 = sig_data[np.random.randint(sig_data.shape[0], size=(n_per_batch)), :]

        sig_target = sig1[:, :sig_len]

        rand_start_idx2 = np.random.randint(sig2.shape[1]-sig_len, size=sig2.shape[0])
        inds2 = tf.cast(rand_start_idx2.reshape(-1,1) + np.arange(sig_len).reshape(1,-1), tf.int32)
        sig_interference = tf.experimental.numpy.take_along_axis(sig2, inds2, axis=1)

        # Interference Coefficient
        rand_gain = np.sqrt(10**(-sinr/10)).astype(np.float32)
        rand_phase = tf.random.uniform(shape=[sig_interference.shape[0],1])
        rand_gain = tf.complex(rand_gain, tf.zeros_like(rand_gain))
        rand_phase = tf.complex(rand_phase, tf.zeros_like(rand_phase))
        coeff = rand_gain * tf.math.exp(1j*2*np.pi*rand_phase)

        sig_mixture = sig_target + sig_interference * coeff

        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig_target)
        all_bits1.append(bits1)

        actual_sinr = get_sinr_db(sig_target, sig_interference * coeff)
        meta_data.append(np.vstack(([rand_gain.numpy().real for _ in range(n_per_batch)], [sinr for _ in range(n_per_batch)], actual_sinr)))

    with tf.device('CPU'):
        all_sig_mixture = tf.concat(all_sig_mixture, axis=0).numpy()
        all_sig1 = tf.concat(all_sig1, axis=0).numpy()
        all_bits1 = tf.concat(all_bits1, axis=0).numpy()

    # changed saving mode --> hdf5
    meta_data = np.concatenate(meta_data, axis=1).T
    
    with h5py.File(os.path.join('dataset', f'TestSet1Example_Dataset_{soi_type}_{interference_sig_type}.h5'), 'w') as hf:
        hf.create_dataset('soi_type', data=soi_type.encode('UTF-8'))
        hf.create_dataset('interference_sig_type', data=interference_sig_type.encode('UTF-8'))
        hf.create_dataset('mixtures', data=all_sig_mixture)
        hf.create_dataset('soi', data=all_sig1)
        hf.create_dataset('bits', data=all_bits1)
        hf.create_dataset('meta_data', data=meta_data)

if __name__ == "__main__":
    soi = [sys.argv[1]] if len(sys.argv) > 1 else ['QPSK', 'OFDMQPSK']
    interference = [sys.argv[2]] if len(sys.argv) > 2 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    for s in soi:
        for i in interference:
            generate_demod_testmixture(s,i)
