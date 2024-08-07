# SOI-specific autoencoder heads for data-driven RF signal separation systems 
Accompanying code for the papers [Extending Data-Driven RF Signal Separation Systems by Signal-Specific Autoencoder Heads](#reference) and [Improving Data-Driven RF Signal Separation with SOI-Matched Autoencoders](https://rfchallenge.mit.edu/wp-content/uploads/2024/02/Lhen_final_paper.pdf).

The papers are based on the [ICASSP 2024 SP Grand Challenge: Data-Driven Signal Separation in Radio Spectrum](https://signalprocessingsociety.org/publications-resources/data-challenges/data-driven-signal-separation-radio-spectrum-icassp-2024).

## Challenge
Click [here](https://rfchallenge.mit.edu/icassp24-single-channel/) for details on the challenge setup.
This GitHub repository is a fork of the challenge organizers' GitHub repository providing the [starter code](https://github.com/RFChallenge/icassp2024rfchallenge).
The RF challenge data can be downloaded manually here: [InterferenceSet](https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0).

## Setup
The code is only tested using Python 3.8.5 and the package versions listed in `requirements.txt`.
Relevant bash commands to set up the code:
```bash
# clone this repository
git clone https://github.com/lukashenneke/RFChallengeSCSS.git
cd RFChallengeSCSS

# install python packages - using Python 3.8 and a virtual environment is recommended to make things work
python install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116

# obtain the dataset
wget -O dataset.zip "https://www.dropbox.com/scl/fi/zlvgxlhp8het8j8swchgg/dataset.zip?rlkey=4rrm2eyvjgi155ceg8gxb5fc4&dl=0"
unzip dataset.zip
rm dataset.zip
```

## Training

The interface for training a signal separation model for a combination of SOI Type and Interference Type is

`python train.py [SOI Type] [Interference Type] -id [Model Identifier] --config [YAML Config File]`

Since the initial training of an autodecoder head depends on a pre-trained WaveNet model and initial training of the autoencoder head should be completed before fine-tuning all model parameters, the order of training should be the following:
```bash
# for example, we are considering signal mixtures of SOI Type OFDMQPSK and Interference Type EMISignal1
# train WaveNet
python train.py OFDMQPSK EMISignal1 -id wavenet --config src/configs/wavenet.yml

# initial training of the SOI-specific autoencoder head
python train.py OFDMQPSK EMISignal1 -id wave-sae --config src/configs/wave-sae.yml

# fine-tuning of all parameters (fine-tuning is indicated by the '_ft' extension of the model identifier)
python train.py OFDMQPSK EMISignal1 -id wave-sae_ft --config src/configs/wave-sae.yml
```

The config files for all experiments studied in [Extending Data-Driven RF Signal Separation Systems by Signal-Specific Autoencoder Heads](#reference) are placed in [src/configs](src/configs).

To re-run all experiments (this will probably take weeks), run:

`python run_training.py`

To only run a subset of the experiments, delete code lines with unwanted experiments in [run_training.py](run_training.py).

## Inference and evaluation

Before starting inference, the evaluation datasets "TestSet1Example", based on the raw interference dataset of TestSet1, have to be generated for all signal mixture scenarios:
```bash
# generate TestSet1Example for specific signal mixture scenario
python testmixture_generator.py [SOI Type] [Interference Type]

# generate TestSet1Example for all 8 scenarios
python testmixture_generator.py
```

To evaluate a trained model, run:
```bash
# evaluate model for specific signal mixture scenario
python inference_and_evaluation.py [Model Identifier] [SOI Type] [Interference Type]

# evaluate model for all 8 scenarios
python inference_and_evaluation.py [Model Identifier]
```

This will save evaluation results to a *.npy file in [outputs](outputs/).
SOI demodulation without interference mitigation can be started with `[Model Identifier] = 'none'`. 
Finally, run 

`python plot_results.py [SOI Type] [Interference Type]`

to plot and print the results. It might be necessary to add your Model Identifier to the model_id list in [plot_results.py](./plot_results.py).

## Reference
COMING SOON