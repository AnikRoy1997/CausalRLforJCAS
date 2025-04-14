# Causally-aware Reinforcement Learning for Joint Communication and Sensing

This repo contains the source code for the implementation of Causally-aware Reinforcement Learning for Joint Communication and Sensing.

## Installation instructions
All experiments are done on Python 3.8.8. You will need dependencies for PyTorch and CUDA-enabled GPU. THe code _may_ run on CPU but the will be slow. To install the dependencies, first set up an environment and then run the following command

```shell
pip install -r requirements.txt
```

You will need MATLAB to visualize the trained beam patterns

## Run an experiment
Folders 'without_interf' and 'with_interf' contain the codes for beam pattern training, placed in deperate folders named after the corresponding beam pattern learning algorithm names. Download the dataset folders from this [google drive link](https://drive.google.com/drive/folders/15HBHfZ42VoOoVyaVGR04idyUKAftHEbg?usp=sharing). Now, for example, if you are running TD3-INVASE for beamforming for 'without_interf' case study, extract and move the contents of the downloaded file 'dataset_withoutinterf.zip' to the directory 'without_interf/TD3INV/'. Follow this step for the other beam pattern learning algorithms (TD3, WDDPG, and ISAC-WDDPG).

### Training
To run an experiment, navigate to the dedicated folder (TD3, WDDPG, and ISAC-WDDPG) and execute the command

```shell
python TD-INVASE_Beamforming.py
```

All training results are stored in '../runs/' folder and can be visualized in tensorboard using the command

```shell
tensorboard --logdir=runs
```

The trained models are saved in the corresponding '../models/' folder.

The trained beam patterns are stored in the '../beams_allscenes/' folder as '.txt' files.

### Beam pattern visualization
To visualize trained beam patterns, execute the below command to create a './mat' in the folder '../beam_codebook_set/'

```shell
python read_beams.py
```

Make sure to provide the appropriate inputs within the read_beams.py, i.e., the number of scenes and the number of beams for which you have trained the model.

Copy the folder contents of '../beam_codebook_set/' and paste it in 'beam_pattern/beam_codebook_set/'. Run the 'beam_pattern/beam_patterns_plotting.m' in MATLAB. The plots will be saved in the '../plots/' folder.
