import os, sys
import numpy as np
import matplotlib.pyplot as plt

all_sinr = np.arange(-30, 0.1, 3)
n_per_batch = 100
SAVE = False

def run_evaluation(testset_identifier, soi_identifier, interference_identifier):
    keep_scores = {}
    keep_mse = {}
    keep_ber = {}
    for soi_type in soi_identifier:
        for interference_sig_type in interference_identifier:
            all_mse, all_ber, all_scores = {}, {}, {}
            for id_string in [
                'none', 'wavenet', 
                'wave-sae', 'wave-sae_ft', 
                'wave-sae_sinr', 'wave-sae_sinr_ft', 
                'wave-sae_16bs', 'wave-sae_16bs_ft', 
                'wave-sae_wo_enc_sc', 'wave-sae_wo_enc_sc_ft', 
                'wave-sae_wo_encdec_sc', 'wave-sae_wo_encdec_sc_ft', 
                'wave-sae_wo_encdec_sc_16bs', 'wave-sae_wo_encdec_sc_16bs_ft', 
                'wave-sae_harddec', 'wave-sae_harddec_ft',
                'wave-sae_lam', 'wave-sae_lam_ft',
                'wave-sae_mse', 'wave-sae_mse_ft', 
                'wave-sae_fo', 'wave-sae_fo_ft', 
                ]:
                try:
                    results = np.load(os.path.join('outputs', f'{id_string}_{testset_identifier}_{soi_type}_{interference_sig_type}_results.npy'))
                except FileNotFoundError:
                    continue
                mse_mean, ber_mean = results[0], results[1]
                
                all_mse[id_string] = mse_mean
                all_ber[id_string] = ber_mean
                mse_score = mse_mean.copy()
                mse_score[mse_score<-50] = -50
                mse_score = round(float(np.mean(mse_score)), 2)
                ber_score = int(-(sum(ber_mean < 1e-2)-1)*3)
                bes_score = ber_mean.copy()
                bes_score[bes_score<1e-6] = 1e-6
                bes_score = round(float(np.mean(10*np.log10(bes_score))), 2)
                all_scores[id_string] = (mse_score, ber_score, bes_score)

            #if len(all_scores) <= 1: continue
            comb = f'{soi_type}_{interference_sig_type}'
            keep_scores[comb] = all_scores
            keep_mse[comb] = all_mse
            keep_ber[comb] = all_ber
            print('===', comb, '===')
            for k,v in all_scores.items(): print(k, v)

            tr_dict = {'abc': 'cef', '123': '456'}
            plt.figure()
            for id_string in all_mse.keys():
                plt.plot(all_sinr, all_mse[id_string], 'x--', label=tr_dict.get(id_string, id_string))
            plt.legend()
            plt.grid()
            plt.gca().set_ylim(top=3)
            plt.xlabel('SINR [dB]')
            plt.ylabel('MSE [dB]')
            plt.title(f'MSE - {soi_type} + {interference_sig_type}')
            plt.show(block=False)

            plt.figure()
            for id_string in all_ber.keys():
                plt.semilogy(all_sinr, all_ber[id_string], 'x--', label=tr_dict.get(id_string, id_string))
            plt.legend()
            plt.grid()
            plt.ylim([1e-4, 1])
            plt.xlabel('SINR [dB]')
            plt.ylabel('BER')
            plt.title(f'BER - {soi_type} + {interference_sig_type}')
            plt.show(block=True)
    final_score = {}
    for k,v in keep_scores.items():
        for kk,vv in v.items():
            if kk not in final_score:
                final_score[kk] = vv
            else:
                final_score[kk] = (round(final_score[kk][0]+vv[0],2), final_score[kk][1]+vv[1], round(final_score[kk][2]+vv[2],2))
    print('=== FINAL SCORES ===')
    for k,v in final_score.items(): print(k, v)
    keep_scores['final_score'] = final_score
    if SAVE:
        import json
        with open('final_scores.json', 'w', encoding='utf-8') as fd:
            json.dump(keep_scores, fd, ensure_ascii=False, indent=4)
        for comb in keep_mse:
            for id in keep_mse[comb]:
                keep_mse[comb][id] = keep_mse[comb][id].tolist()
        with open('final_mse.json', 'w', encoding='utf-8') as fd:
            json.dump(keep_mse, fd, ensure_ascii=False, indent=4)
        for comb in keep_ber:
            for id in keep_ber[comb]:
                keep_ber[comb][id] = keep_ber[comb][id].tolist()
        with open('final_ber.json', 'w', encoding='utf-8') as fd:
            json.dump(keep_ber, fd, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    soi_type = [sys.argv[1]] if len(sys.argv) > 1 else ['QPSK', 'OFDMQPSK']
    interference_sig_type = [sys.argv[2]] if len(sys.argv) > 2 else ['EMISignal1', 'CommSignal2', 'CommSignal3', 'CommSignal5G1']
    testset_identifier = sys.argv[3] if len(sys.argv) > 3 else 'TestSet1Example'

    run_evaluation(testset_identifier, soi_type, interference_sig_type)