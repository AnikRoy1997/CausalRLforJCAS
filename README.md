# Causally-aware Reinforcement Learning for Joint Communication and Sensing

This repo contains the source code for the implementation of Causally-aware Reinforcement Learning for Joint Communication and Sensing.

## Installation instructions
All experiments are done on Python 3.8.8. You will need dependencies for PyTorch and CUDA-enabled GPU. THe code _may_ run on CPU but the will be slow. To install the dependencies, first set up an environment and then run the following command
```pip install -r requirements.txt```

You will need MATLAB to visualize the trained beam patterns

## Run an experiment
The repo contains three main forders: 'no_ant_tilt' for 'without antenna tilt' case study, 'ant_tilt' for 'without antenna tilt' case study, and 'with_interference' for interference-aware beamforming case study. Within each main folders, there are folders named with the corresponding beam pattern learning algorithm. 

### Training
To run an experiment, navigate to the dedicated experiment folder and execute the command
```python TD-INVASE_Beamforming.py```

All training results can be visualized in tensorboard using the command
```tensorboard --logdir=runs```

The trained models are saved in the corresponding '../models/' folder.

### Beam pattern visualization
The trained beam patterns are stored in the '../beams_allscenes/' folder as '.txt' files. Execute the below command to create a './mat' in the folder '../beam_codebook_set/'
```python read_beams.py```
Make sure to provide the appropriate inputs within the read_beams.py, i.e., the number of scenes and the number of beams for which you have trained the model.

Copy the folder contents of '../beam_codebook_set/' and paste it in 'beam_pattern/beam_codebook_set/'. Run the 'beam_patterns_plotting.m' in MATLAB. The plots will be saved in the '../plots/' folder.
