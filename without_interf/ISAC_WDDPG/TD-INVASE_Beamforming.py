ENV_NAME = 'env_CB'  #env_CB
alias = 'TD_INVASE'
RED_ACTION_DIM = 100
import gym
import os
from gym.envs.registration import register
print('\n now evaluating: \n       ', ENV_NAME)

from torch.utils.tensorboard import SummaryWriter
# import socket
# from datetime import datetime

# current_time = datetime.now().strftime("%b%d_%H-%M-%S")
# log_dir = os.path.join(
#     "/home/anirban/anik/Causal_JCAS/JCAS/tmlcn_review/All_results/runs", current_time + "_" + socket.gethostname()
# )
writer = SummaryWriter()


import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import utils
import TD3_INVASE_TD
import TD3
import DDPG

from DataPrep import dataPrep
from clustering import KMeans_only
from function_lib import bf_gain_cal, corr_mining
import time
import math
import copy
import pickle
import h5py as h5
from scipy.optimize import linear_sum_assignment

options = {
    'gpu_idx': 0,
    'num_ant': 32,
    'num_bits': 4,
    'ant_sep': 0.5,
    'wavelength': 0.005,
    'num_NNs': 1,  # codebook size
    'num_NNs_sensing': 1,   #Anik
    'overall_episodes': 1,
    'ch_sample_ratio': 0.5,
    'num_loop': 1,  # outer loop
    'target_update': 3,
    'path': './O2_channel_dataset/',
    'scenes': 4,  #Number of scenes (dynamic scenario only) Anik 22-08-23
    'clustering_mode': 'random',
}

def eval_policy(policy, eval_env, scene, eval_episodes=10):
    avg_reward = 0.
    avg_bf_gain = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = torch.zeros((1, options['num_ant'])).float().cpu()
        ep_reward = 0.
        ep_bf_gain = 0.
        episode_timestep = 0.
        while not done:
            episode_timestep += 1
            action = policy.select_action(np.array(state))
            action = torch.from_numpy(action).float().reshape(1,-1).cpu()
            if RED_ACTION_DIM>0:
                step_inputs={
                    'action': action[:,:-RED_ACTION_DIM],
                    'episode_timestep': episode_timestep,
                    'eval_episodes': eval_episodes,
                    'scene': scene
                }
            else:
                step_inputs={
                    'action': action[:,:],
                    'episode_timestep': episode_timestep,
                    'eval_episodes': eval_episodes,
                    'scene': scene
                }
            next_state, reward, bf_gain, done, _ = eval_env.step(step_inputs)
            reward = float(reward[0,0])
            state = next_state
            ep_bf_gain += 10*math.log10(bf_gain)
            ep_reward += reward
        avg_bf_gain += ep_bf_gain/episode_timestep
        avg_reward += ep_reward/episode_timestep
    avg_bf_gain /= eval_episodes
    avg_reward /= eval_episodes
#     avg_reward = avg_reward.astype(float)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: Avg_reward={avg_reward:.3f} Avg_BF_gain:{avg_bf_gain:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_bf_gain

def eval_policy_sensing(policy, eval_env, scene, eval_episodes=10):
    avg_reward = 0.
    avg_bf_gain = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = torch.zeros((1, options['num_ant'])).float().cpu()
        ep_reward = 0.
        ep_bf_gain = 0.
        episode_timestep = 0.
        while not done:
            episode_timestep += 1
            action = policy.select_action(np.array(state))
            action = torch.from_numpy(action).float().reshape(1,-1).cpu()
            if RED_ACTION_DIM>0:
                step_inputs={
                    'action': action[:,:-RED_ACTION_DIM],
                    'episode_timestep': episode_timestep,
                    'eval_episodes': eval_episodes,
                    'num_NNs': options['num_NNs'],
                    'scene': scene
                }
            else:
                step_inputs={
                    'action': action[:,:],
                    'episode_timestep': episode_timestep,
                    'eval_episodes': eval_episodes,
                    'num_NNs': options['num_NNs'],
                    'scene': scene
                }
            next_state, reward, bf_gain, done, _ = eval_env.step_sensing(step_inputs)
            reward = float(reward[0,0])
            state=next_state
            ep_bf_gain += 10*math.log10(bf_gain)
            ep_reward += reward
        avg_bf_gain += ep_bf_gain/episode_timestep
        avg_reward += ep_reward/episode_timestep

    avg_bf_gain /= eval_episodes
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: Avg_reward={avg_reward:.3f} Avg_BF_gain:{avg_bf_gain:.3f}")
    print("---------------------------------------")
    return avg_reward, avg_bf_gain

if not os.path.exists('beams/'):
    os.mkdir('beams/')

#spec = env.action_space
state_dim = options['num_ant']
action_dim = options['num_ant'] + RED_ACTION_DIM
max_action = np.pi

grnd_truth = np.concatenate([np.ones((options['num_ant'],1)),np.zeros((RED_ACTION_DIM,1))]).reshape(1,-1).flatten()

args_policy_noise = 0.2
args_noise_clip = 0.5
args_policy_freq = 2
args_max_timesteps = 200000  # 50000, 200000
args_expl_noise = 0.1
replay_memory_size = 50000  #8192
args_batch_size = 1024
args_eval_freq = 1000
args_start_timesteps = 50000  # 10000, 40000

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": 0.99,
    "tau": 0.005
}
kwargs["policy_noise"] = args_policy_noise * max_action
kwargs["noise_clip"] = args_noise_clip * max_action     #comment these out while using ddpg
kwargs["policy_freq"] = args_policy_freq
# replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

veh_loc = np.genfromtxt('O2_veh_dataset/paths_O2_dyn.csv', delimiter=',')
# print(np.size(veh_loc[0,:]))
# print(veh_loc[0,2])   #veh_loc[<scene number>,<x or y coord for different veh>]
with h5.File('BS_loc.mat', 'r') as f:
    fields = [k for k in f.keys()]
    nested = [k for k in f[fields[0]]]
    BS_loc = np.squeeze(np.array(nested))
# print(BS_loc[0])

for scene in range(options['scenes']):
    if not os.path.exists('beams_allscenes/beams_'+str(scene)+'/'):   #Anik 22-08-23
        os.mkdir('beams_allscenes/beams_'+str(scene)+'/')   #Anik 22-08-23
        for beam_id in range(options['num_NNs']):
            open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(beam_id) + '_max.txt', 'ab')

    with h5.File('O2_loc_dataset/loc_dataset_'+str(scene)+'.mat', 'r') as f:
        fields = [k for k in f.keys()]
        nested = [k for k in f[fields[0]]]
        locs = np.squeeze(np.array(nested))
        # print(np.size(locs[0]))  #locs[<x or y or z coord (0,1,2)>,<user number>]

    d = np.zeros([np.size(locs[0]),int(np.size(veh_loc[0,:])/2)])
    #find vehicle distance closest to the user grid
    for i in range(np.size(locs[0])):  #user number
        for j  in range(0, np.size(veh_loc[0,:]), 2):   #vehicle number
            d[i,int(j/2)] = np.sqrt(np.square(locs[0,i]-veh_loc[scene,j]) + np.square(locs[1,i]-veh_loc[scene,j+1]))
    # print(np.shape(d))        
    min_dist = np.min(d)
    min_vehid = 0.0
    # print(min_dist)
    for i in range(np.size(locs[0])):  #user number
        for j  in range(0, np.size(veh_loc[0,:]), 2):   #vehicle number
            if d[i,int(j/2)] == min_dist:
                min_vehid = int(j/2)
    # print(min_vehid)
                
    #angle with nearest base station
    dx = veh_loc[scene,min_vehid] - BS_loc[0]
    dy = veh_loc[scene,min_vehid+1] - BS_loc[1]
    theta = math.atan2(dy,dx)
    print(theta*(180/np.pi))

    # ch = dataPrep(options['path']+'ch_dataset_'+str(scene)+'.mat')   #Anik 22-08-23
    ch = dataPrep(options['path']+'/ch_dataset_'+str(scene)+'.mat')
    ch = np.concatenate((ch[:, :options['num_ant']],
                        ch[:, int(ch.shape[1] / 2):int(ch.shape[1] / 2) + options['num_ant']]), axis=1)
    print(ch.shape)
    register(
        id='env_CB',
        entry_point='env_CB:env_CB',  # Replace 'custom_env' with the name of your Python file and 'CustomEnv' with your environment class name
    )
    env_list = []
    #Envorinment list for DRLs:
    for beam_id in range(options['num_NNs']+options['num_NNs_sensing']): 
        env = gym.make(ENV_NAME,ch=ch,num_ant=options['num_ant'],num_bits=options['num_bits'],idx=beam_id,scene=scene,red_act=RED_ACTION_DIM,theta=theta,d=options['ant_sep'],wavelength=options['wavelength'],options=options)
        env_list.append(env)  #Anik 22-08-23
        torch.manual_seed(0)
        np.random.seed(0)

    with torch.cuda.device(options['gpu_idx']):
        u_classifier, sensing_beam = KMeans_only(ch, options['num_NNs'], n_bit=options['num_bits'], n_rand_beam=30)
        np.save('sensing_beam.npy', sensing_beam)
        sensing_beam = torch.from_numpy(sensing_beam).float().cuda()

        filename = 'kmeans_model.sav'
        pickle.dump(u_classifier, open(filename, 'wb'))

        # Quantization settings
        options['num_ph'] = 2 ** options['num_bits']
        options['multi_step'] = torch.from_numpy(
            np.linspace(int(-(options['num_ph'] - 2) / 2),
                        int(options['num_ph'] / 2),
                        num=options['num_ph'],
                        endpoint=True)).type(dtype=torch.float32).reshape(1, -1).cuda()
        options['pi'] = torch.tensor(np.pi).cuda()
        options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
        options['ph_table'].cuda()
        options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)
        options['ph_table_rep_redact'] = options['ph_table'].repeat(action_dim, 1)

        num_episodes_comm = np.ones((1,options['num_NNs']))
        num_episodes_sensing = np.ones((1,options['num_NNs_sensing']))

        for sample_id in range(options['num_loop']):
            # ---------- Sampling ---------- #
            n_sample = int(ch.shape[0] * options['ch_sample_ratio'])
            ch_sample_id = np.random.permutation(ch.shape[0])[0:n_sample]
            ch_sample = torch.from_numpy(ch[ch_sample_id, :]).float().cuda()
            # ch_sample = torch.from_numpy(ch[ch_sample_id, :]).float().cpu()

            # ---------- Clustering ---------- #
            start_time = time.time()

            bf_mat_sample = bf_gain_cal(sensing_beam, ch_sample)
            # print("Clustering -1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            f_matrix = corr_mining(bf_mat_sample)
            f_matrix_np = torch.Tensor.cpu(f_matrix).numpy()
            # print("Clustering 0 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            labels = u_classifier.predict(np.transpose(f_matrix_np).astype(float))

            # print("Clustering 1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()

            user_group = []  # order: clusters
            ch_group = []  # order: clusters
            for ii in range(options['num_NNs']):
                user_group.append(np.where(labels == ii)[0].tolist())
                ch_group.append(ch_sample[user_group[ii], :])

            print("Clustering 2 uses %s seconds." % (time.time() - start_time))

            # ---------- Assignment ---------- #
            start_time = time.time()

            # best_state matrix
            best_beam_mtx = torch.zeros((options['num_NNs'], 2 * options['num_ant'])).float().cuda()
            # best_beam_mtx = torch.zeros((options['num_NNs'], 2 * options['num_ant'])).float().cpu()
            for pp in range(options['num_NNs']):
                best_beam_mtx[pp, :] = env_list[pp].best_bf_vec
            gain_mtx = bf_gain_cal(best_beam_mtx, ch_sample)  # (n_beam, n_user)
            for ii in range(options['num_NNs']):
                if ii == 0:
                    cost_mtx = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                else:
                    sub = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                    cost_mtx = torch.cat((cost_mtx, sub), dim=1)
            cost_mtx = -torch.Tensor.cpu(cost_mtx).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_mtx)
            assignment_record = dict(zip(row_ind.tolist(), col_ind.tolist()))  # key: network, value: cluster
            print(assignment_record)
            for ii in range(options['num_NNs']):
                env_list[ii].ch = ch_group[assignment_record[ii]]

            print("Assignment uses %s seconds." % (time.time() - start_time))
            

            for beam_id in range(options['num_NNs']):

                policy = TD3_INVASE_TD.TD3(**kwargs)
                # policy = TD3.TD3(**kwargs)
                # policy = DDPG.DDPG(**kwargs)
                replay_memory = []

                # Evaluate untrained policy
                evaluations = [eval_policy(policy, env_list[beam_id], scene)]
                state, done = env_list[beam_id].reset(), False
                state = torch.zeros((1, options['num_ant'])).float().cuda()
                episode_reward = 0
                episode_bfgain = 0
                episode_data_rate = 0
                episode_tpr = 0
                episode_fdr = 0
                episode_timesteps = 0
                episode_num = 0
                counter = 0
                msk_list = []        
                temp_curve = [eval_policy(policy, env_list[beam_id], scene)]
                temp_val = []

                start_time = time.time()

                for t in range(int(args_max_timesteps)):
                    episode_timesteps += 1
                    counter += 1
                    # Select action randomly or according to policy
                    if t < args_start_timesteps:
                        action = np.random.uniform(-max_action, max_action, action_dim) 
                    else:
                        if np.random.uniform(0,1) < 0.0:
                            action = np.random.uniform(-max_action, max_action, action_dim)
                        else:
                            action = (
                                policy.select_action(np.array(state.cpu()))
                                + np.random.normal(0, max_action * (1 - t/args_max_timesteps), size=action_dim)
                            ).clip(-max_action, max_action)

                    action = torch.from_numpy(action).reshape((1,-1)).float().cuda()
                    reward_pred, bf_gain_pred, action_quant_pred, next_state_pred = env_list[beam_id].get_reward(scene, action[:,:-RED_ACTION_DIM])
                    reward_pred = torch.from_numpy(reward_pred).float().cpu()
                    #action quantization
                    mat_dist = torch.abs(action.reshape(action_dim,1) - options['ph_table_rep_redact'])
                    action_quant = options['ph_table_rep_redact'][range(action_dim), torch.argmin(mat_dist, dim=1)].reshape(1,-1)
                    action = action_quant.reshape((1,-1)).float().cuda()
                    if RED_ACTION_DIM>0:
                        step_inputs={
                            'action': action[:,:-RED_ACTION_DIM],
                            'episode_timestep': episode_timesteps,
                            'eval_episodes': 200,
                            'scene': scene
                        }
                    else:
                        step_inputs={
                            'action': action[:,:],
                            'episode_timestep': episode_timesteps,
                            'eval_episodes': 200,
                            'scene': scene
                        }
                    next_state, reward, bf_gain, done, _ = env_list[beam_id].step(step_inputs)
                    reward = torch.from_numpy(reward).float().cpu()
                    done_bool = float(done) if episode_timesteps < env_list[beam_id].max_episode_steps else 0

                    # replay_buffer.add(state.cpu(), action.cpu(), next_state.cpu(), reward, done_bool)
                    replay_memory.append((state.cpu(), action.cpu(), next_state.cpu(), reward, done_bool))
                    replay_memory.append((state.cpu(), action.cpu(), next_state_pred.cpu(), reward_pred, done_bool))
                    while len(replay_memory) > replay_memory_size:
                        replay_memory.pop(0)

                    reward = float(reward[0,0])
                    state = next_state
                    episode_reward += reward
                    episode_bfgain += 10*math.log10(bf_gain)

                    if t >= args_start_timesteps:
                        '''TD3'''
                        Lmd = t/args_max_timesteps * 0.1
                        Thr = 0.5*(1 - t/args_max_timesteps)
                        policy.train(replay_memory, args_batch_size, Lmd, Thr)
                        # policy.train(replay_memory, args_batch_size)  #for TD3 and DDPG

                        sel_action_prob = policy.select_action_invase(np.array(state.cpu()), np.array(action.cpu()))
                        sel_action_tot = 1.*(sel_action_prob>0.5)
                        TPR_Num = np.zeros((action_dim,1))
                        TPR_Den = np.zeros((action_dim,1))
                        FDR_Num = np.zeros((action_dim,1))
                        FDR_Den = np.zeros((action_dim,1))
                        TPR_Num = np.sum(np.multiply(sel_action_tot.reshape(1,-1).flatten(),grnd_truth))
                        TPR_Den = np.sum(grnd_truth)
                        FDR_Num = np.sum(np.multiply(sel_action_tot.reshape(1,-1).flatten(),(1-grnd_truth)))
                        FDR_Den = np.sum(sel_action_tot.reshape(1,-1).flatten())
                        TPR_val = 100 * (float(TPR_Num)/float(TPR_Den))
                        # FDR_val = 100 * (float(FDR_Num)/float(FDR_Den))
                        episode_tpr += TPR_val
                        # episode_fdr += FDR_val

                    # Train agent after collecting sufficient data
                    if done:
                        print(f"Beam id: {beam_id} Scene: {scene} Total T: {t+1} Overall_Episodes: {options['overall_episodes']} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {(episode_reward/episode_timesteps):.3f} BF_gain: {(episode_bfgain/episode_timesteps):.3f}")
                        msk_list.append(episode_reward)
                        state, done = env_list[beam_id].reset(), False
                        state = torch.zeros((1, options['num_ant'])).float().cuda()
                        # writer.add_scalar("True Positive Rate (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+")",(episode_tpr/episode_timesteps),num_episodes_comm[0,beam_id])
                        # writer.add_scalar("False Discovery Rate (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+")",(episode_fdr/episode_timesteps),num_episodes_comm[0,beam_id])
                        episode_reward = 0
                        episode_bfgain = 0
                        # episode_data_rate = 0
                        episode_tpr = 0
                        episode_fdr = 0
                        episode_timesteps = 0
                        episode_num += 1
                        # num_episodes_comm[0,beam_id] += 1
                        options['overall_episodes'] += 1 

                    # Evaluate episode
                    if (t + 1) % args_eval_freq == 0:
                        # evaluations.append(eval_policy(policy, env_list[beam_id], num_episodes_comm[0,beam_id], beam_id, scene))
                        avg_reward, avg_bf_gain = eval_policy(policy, env_list[beam_id], scene)
                        writer.add_scalar("Reward (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+")",avg_reward,num_episodes_comm[0,beam_id])
                        writer.add_scalar("Gain of Target User (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+")",avg_bf_gain,num_episodes_comm[0,beam_id])
                        num_episodes_comm[0,beam_id] += 1
                        # print('recent Evaluation:',evaluations[-1])
                        policy.save(f"./models/Scene_"+str(scene)+"_Comm_"+str(beam_id))
    #                     np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)
                writer.add_scalar("Training time: (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+")", (time.time()-start_time), beam_id)
                print("Training time: (Scene: "+str(scene)+", CommBeam: "+str(beam_id)+"): "+str(time.time()-start_time))

        for beam_id in range(options['num_NNs'], options['num_NNs']+options['num_NNs_sensing']):
            # replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
            policy = TD3_INVASE_TD.TD3(**kwargs)
            # policy = TD3.TD3(**kwargs)
            # policy = DDPG.DDPG(**kwargs)
            replay_memory = []

            # Evaluate untrained policy
            evaluations = [eval_policy_sensing(policy, env_list[beam_id], scene)]

            state, done = env_list[beam_id].reset(), False
            state = torch.zeros((1, options['num_ant'])).float().cuda()
            episode_reward = 0
            episode_bfgain = 0
            episode_tpr = 0
            episode_fdr = 0
            episode_timesteps = 0
            episode_num = 0
            counter = 0
            msk_list = []        
            temp_curve = [eval_policy_sensing(policy, env_list[beam_id], scene)]
            temp_val = []

            start_time = time.time()
            
            for t in range(int(args_max_timesteps)):
                episode_timesteps += 1
                counter += 1
                # Select action randomly or according to policy
                if t < args_start_timesteps:
                    action = np.random.uniform(-max_action, max_action, action_dim) 
                else:
                    if np.random.uniform(0,1) < 0.0:
                        action = np.random.uniform(-max_action, max_action, action_dim)
                    else:
                        action = (
                            policy.select_action(np.array(state.cpu()))
                            + np.random.normal(0, max_action * (1 - t/args_max_timesteps), size=action_dim)
                        ).clip(-max_action, max_action)
                
                action = torch.from_numpy(action).reshape((1,-1)).float().cuda()
                reward_pred, bf_gain_pred, action_quant_pred, next_state_pred = env_list[beam_id].get_reward_sensing(scene, action[:,:-RED_ACTION_DIM])
                reward_pred = torch.from_numpy(reward_pred).float().cpu()
                #action quantization
                mat_dist = torch.abs(action.reshape(action_dim,1) - options['ph_table_rep_redact'])
                action_quant = options['ph_table_rep_redact'][range(action_dim), torch.argmin(mat_dist, dim=1)].reshape(1,-1)
                action = action_quant.reshape((1,-1)).float().cuda()
                if RED_ACTION_DIM>0:
                    step_inputs={
                        'action': action[:,:-RED_ACTION_DIM],
                        'episode_timestep': episode_timesteps,
                        'eval_episodes': 200,
                        'scene': scene
                    }
                else:
                    step_inputs={
                        'action': action[:,:],
                        'episode_timestep': episode_timesteps,
                        'eval_episodes': 200,
                        'scene': scene
                    }
                next_state, reward, bf_gain, done, _ = env_list[beam_id].step_sensing(step_inputs)
                reward = torch.from_numpy(reward).float().cpu()
                done_bool = float(done) if episode_timesteps < env_list[beam_id].max_episode_steps else 0
                # replay_buffer.add(state.cpu(), action.cpu(), next_state.cpu(), reward, done_bool)
                replay_memory.append((state.cpu(), action.cpu(), next_state.cpu(), reward, done_bool))
                replay_memory.append((state.cpu(), action.cpu(), next_state_pred.cpu(), reward_pred, done_bool))
                while len(replay_memory) > replay_memory_size:
                    replay_memory.pop(0)

                reward = float(reward[0,0])
                state = next_state
                episode_reward += reward
                episode_bfgain += 10*math.log10(bf_gain)

                if t >= args_start_timesteps:
                    '''TD3'''
                    Lmd = t/args_max_timesteps * 0.1
                    Thr = 0.5*(1 - t/args_max_timesteps)
                    policy.train(replay_memory, args_batch_size, Lmd, Thr)
                    # policy.train(replay_memory, args_batch_size)  #for TD3 and DDPG

                    sel_action_prob = policy.select_action_invase(np.array(state.cpu()), np.array(action.cpu()))
                    sel_action_tot = 1.*(sel_action_prob>0.5)
                    TPR_Num = np.zeros((action_dim,1))
                    TPR_Den = np.zeros((action_dim,1))
                    FDR_Num = np.zeros((action_dim,1))
                    FDR_Den = np.zeros((action_dim,1))
                    TPR_Num = np.sum(np.multiply(sel_action_tot.reshape(1,-1).flatten(),grnd_truth))
                    TPR_Den = np.sum(grnd_truth)
                    FDR_Num = np.sum(np.multiply(sel_action_tot.reshape(1,-1).flatten(),(1-grnd_truth)))
                    FDR_Den = np.sum(sel_action_tot.reshape(1,-1).flatten())
                    TPR_val = 100 * (float(TPR_Num)/float(TPR_Den))
                    # FDR_val = 100 * (float(FDR_Num)/float(FDR_Den))
                    episode_tpr += TPR_val
                    # episode_fdr += FDR_val

                # Train agent after collecting sufficient data
                if done:
                    print(f"Sensing Beam id: {beam_id} Scene: {scene} Total Time: {t+1} Overall Episodes: {options['overall_episodes']} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {(episode_reward/episode_timesteps):.3f} BF_gain: {(episode_bfgain/episode_timesteps):.3f}")
                    msk_list.append(episode_reward)
                    state, done = env_list[beam_id].reset(), False
                    state = torch.zeros((1, options['num_ant'])).float().cuda()
                    # writer.add_scalar("True Positive Rate (Scene: "+str(scene)+", SensingBeam: "+str(beam_id)+")",(episode_tpr/episode_timesteps),num_episodes_sensing[0,beam_id-options['num_NNs']])
                    # writer.add_scalar("False Discovery Rate (Scene: "+str(scene)+", sensingBeam: "+str(beam_id)+")",(episode_fdr/episode_timesteps),num_episodes_sensing[0,beam_id-options['num_NNs']]) 
                    episode_reward = 0
                    episode_bfgain = 0
                    episode_tpr = 0
                    episode_fdr = 0
                    episode_timesteps = 0
                    episode_num += 1
                    options['overall_episodes'] += 1 

                # Evaluate episode
                if (t + 1) % args_eval_freq == 0:
                    # evaluations.append(eval_policy_sensing(policy, env_list[beam_id], num_episodes_sensing[0,beam_id-options['num_NNs']], beam_id, scene))
                    avg_reward, avg_bf_gain = eval_policy_sensing(policy, env_list[beam_id], scene)
                    writer.add_scalar("Reward (Scene: "+str(scene)+", SensingBeam: "+str(beam_id)+")",avg_reward,num_episodes_sensing[0,beam_id-options['num_NNs']])
                    writer.add_scalar("Gain of Target User (Scene: "+str(scene)+", SensingBeam: "+str(beam_id)+")",avg_bf_gain,num_episodes_sensing[0,beam_id-options['num_NNs']])
                    num_episodes_sensing[0,beam_id-options['num_NNs']] += 1
                    # print('recent Evaluation:',evaluations[-1])
                    policy.save(f"./models/Scene_"+str(scene)+"_Sense_"+str(beam_id))
            writer.add_scalar("Training time: (Scene: "+str(scene)+", SensingBeam: "+str(beam_id)+")", (time.time()-start_time), beam_id)
            print("Training time: (Scene: "+str(scene)+", SensingBeam: "+str(beam_id)+"): "+str(time.time()-start_time))
    
    # plt.plot(msk_list)
    # plt.savefig('Reward_TD3_INVASE_TD.png')