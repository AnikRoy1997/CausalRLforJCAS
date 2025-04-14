import os
import numpy as np
import scipy.io as scio

num_ant = 32
num_beam = 2
num_scene = 4  #Anik 22-08-23

if not os.path.exists('beam_codebook_set/'):   #Anik 22-08-23
    os.mkdir('beam_codebook_set/')   #Anik 22-08-23

for scene in range(num_scene):   #Anik 22-08-23
    if not os.path.exists('./beams_allscenes/beams_'+str(scene)+'/'):
        os.mkdir('./beams_allscenes/beams_'+str(scene)+'/')
    path = './beams_allscenes/beams_'+str(scene)+'/'   #Anik 22-08-23
    results = np.empty((num_beam, 2*num_ant))
    for beam_id in range(num_beam):
        fname = 'beams_' + str(beam_id) + '_max.txt'
        with open(path + fname, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            results[beam_id, :] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

    results = (1 / np.sqrt(num_ant)) * (results[:, ::2] + 1j * results[:, 1::2])

    # with open('performance.txt', 'r') as f:
    #     lines = f.readlines()
    
    # itr_val, perfor_val = lines[::2], lines[1::2]

    # # print(results[beam_id, :])
    # with open('perf_TD3.txt','w') as f1:
    #     for odd_line in perfor_val:
    #         f1.write(odd_line)

    # with open('itr_TD3.txt','w') as f2:
    #     for even_line in itr_val:
    #         f2.write(even_line)
    # print(results[beam_id, :])
    scio.savemat('./beam_codebook_set/beam_codebook_'+str(scene)+'.mat', {'beams': results})  #Anik 22-08-23
