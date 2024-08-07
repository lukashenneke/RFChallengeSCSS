import sys
from train import main

soi = ['QPSK', 'OFDMQPSK']
interference = ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
# ATTENTION: training of all models takes weeks
id_yml = [
    ('wavenet', 'src/configs/wavenet.yml'), # baseline wavenet
    ('wave-sae', 'src/configs/wave-sae.yml'), ('wave-sae_ft', 'src/configs/wave-sae.yml'), # challenge setup, initial training + fine-tunnig
    ('wave-sae_sinr', 'src/configs/wave-sae_sinr.yml'), ('wave-sae_sinr_ft', 'src/configs/wave-sae_sinr.yml'), # full SINR range
    ('wave-sae_16bs', 'src/configs/wave-sae_16bs.yml'), ('wave-sae_16bs_ft', 'src/configs/wave-sae_16bs.yml'), # batch size 16
    ('wave-sae_wo_enc_sc', 'src/configs/wave-sae_wo_enc_sc.yml'), ('wave-sae_wo_enc_sc_ft', 'src/configs/wave-sae_wo_enc_sc.yml'), # no encoder scaling
    ('wave-sae_wo_encdec_sc', 'src/configs/wave-sae_wo_encdec_sc.yml'), ('wave-sae_wo_encdec_sc_ft', 'src/configs/wave-sae_wo_encdec_sc.yml'), # no scaling
    ('wave-sae_wo_encdec_sc_16bs', 'src/configs/wave-sae_wo_encdec_sc_16bs.yml'), ('wave-sae_wo_encdec_sc_16bs_ft', 'src/configs/wave-sae_wo_encdec_sc_16bs.yml'), # no scaling + batch size 16
    ('wave-sae_harddec', 'src/configs/wave-sae_harddec.yml'), ('wave-sae_harddec_ft', 'src/configs/wave-sae_harddec.yml'), # hard bit decision @ decoder
    ('wave-sae_lam', 'src/configs/wave-sae_lam.yml'), ('wave-sae_lam_ft', 'src/configs/wave-sae_lam.yml'), # lambda_ber = 10
    ('wave-sae_mse', 'src/configs/wave-sae_mse.yml'), ('wave-sae_mse_ft', 'src/configs/wave-sae_mse.yml'), # MSE-based BER-loss function
    ('wave-sae_fo', 'src/configs/wave-sae_fo.yml'), ('wave-sae_fo_ft', 'src/configs/wave-sae_fo.yml'), # frequency offset augmentation
] 

if __name__ == '__main__':
    tmp = sys.argv
    for id, yml in id_yml:
        for i in interference:
            for s in soi:
                print(id, s, i)
                sys.argv = tmp + [s, i, '-id', id, '--config', yml]
                try:
                    main()
                except SystemExit as e:
                    pass
                except RuntimeError as r:
                    print(' ')
                    print(r)
                    pass