import torch
import numpy as np
import tensorflow as tf
from tqdm import tqdm

class MixtureDataset(torch.utils.data.Dataset):

    def __init__(self, soi_gen_fn, sig_inf, sig_len, N, sinr_range, freqoffset_std=0, return_bits=False, fix=False):
        self.soi_gen_fn = soi_gen_fn
        self.sig_inf = sig_inf
        self.sig_len = sig_len
        self.N = N
        self.sinr = sinr_range
        self.fo_std = freqoffset_std
        self.return_bits = return_bits
        self.fix = False
        if fix:
            self.data = []
            for ii in tqdm(range(N), desc='Preparing fixed data'):
                self.data.append(self.__getitem__(ii))
        self.fix = fix

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        if self.fix: return self.data[index]
        with tf.device('CPU'):
            sig_target, _, bits, _ = self.soi_gen_fn(1, self.sig_len)
            sig_target = sig_target[0, :self.sig_len].numpy()
        sig_interference = self.sig_inf[np.random.randint(self.sig_inf.shape[0]), :]
        start_idx = np.random.randint(sig_interference.shape[0]-self.sig_len)
        sig_interference = sig_interference[start_idx:start_idx+self.sig_len]

        # carrier frequency offset
        if self.fo_std > 0:
            cfo = np.random.randn()*self.fo_std
            sig_interference *= np.exp(1j*2*np.pi*np.arange(self.sig_len)*cfo)

        # Interference Coefficient
        sinr = self.sinr[0] + (self.sinr[1] - self.sinr[0])*np.random.rand()
        coeff = np.sqrt(10**(-sinr/10)) * np.exp(1j*2*np.pi*np.random.rand())

        sig_mixture = sig_target + sig_interference * coeff
        if self.return_bits:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target), np.squeeze(bits.numpy())
        else:
            return torch.from_numpy(sig_mixture), torch.from_numpy(sig_target)