import gym #Anik
from gym import spaces #Anik
import os, math
import torch
import numpy as np
import random
import scipy
import scipy.special as sp
from scipy.integrate import quad
from scipy.special import iv


class env_CB(gym.Env):

    def __init__(self, ch, num_ant, num_bits, idx, scene, red_act, theta, theta_p1, theta_p2, d, wavelength, options):
#         super(env_CB, self).__init__()
        self.idx = idx
        self.num_ant = num_ant
        self.num_bits = num_bits
        self.theta = theta  #Anik
        self.theta_p1 = theta_p1  #Anik
        self.theta_p2 = theta_p2  #Anik
        self.ant_sep = d  #Anik
        self.wavelength = wavelength
        self.scene = scene  #Anik 22-08-23
        self.red_act = red_act
        self.cb_size = 2 ** 4
        self.codebook = self.codebook_gen()
        self.steering_t_vec = self.steering_vec_gen(theta).float().cuda()  #Anik-steering vec gen
        self.steering_i1_vec = self.steering_vec_gen(theta_p1).float().cuda()  #Anik-steering vec gen
        self.steering_i2_vec = self.steering_vec_gen(theta_p2).float().cuda()  #Anik-steering vec gen
        self.ch = torch.from_numpy(ch).float().cuda()
        self.state = torch.zeros((1, self.num_ant)).float().cuda()
        # self.ch = torch.from_numpy(ch).float().cpu()
        # self.state = torch.zeros((1, self.num_ant)).float().cpu()
        self.bf_vec = self.init_bf_vec()
        self.previous_gain = 0
        self.previous_gain_pred = 0
        self.previous_sir = 0
        self.previous_scr = 0
        self.previous_sensing_gain = 0
        
        self.th_step = 0.01
        self.threshold = torch.tensor([0]).float().cuda()
        self.threshold_sir = torch.tensor([0]).float().cuda()
        self.threshold_sensing = torch.tensor([0]).float().cuda()
        self.threshold_scr = torch.tensor([0]).float().cuda()
        # self.threshold = torch.tensor([0]).float().cpu()
        self.count = 1
        self.record_freq = 10
        self.record_decay_th = 1000
        self.bf_gain_t_ach = torch.tensor([0]).float().cuda()
        self.interf_gain_1_ach = torch.tensor([0]).float().cuda()
        self.interf_gain_2_ach = torch.tensor([0]).float().cuda()
        self.bf_gain_t_ach_sensing = torch.tensor([0]).float().cuda()
        self.interf_gain_1_ach_sensing = torch.tensor([0]).float().cuda()
        self.interf_gain_2_ach_sensing = torch.tensor([0]).float().cuda()
        # self.achievement = torch.tensor([0]).float().cpu()
        self.gain_record = [np.array(0)]
        self.sensing_gain_record = [np.array(0)]
        self.N_count = 1
        self.best_bf_vec = self.init_best()
        self.best_bf_sensing_vec = self.init_best()
        self.opt_bf_gain()
        self.options = options
        self.max_episode_steps = 100   #Anik

    def step(self, step_input):  # input_action: (1, num_ant), rep: phase vector
        input_action = step_input['action']
        episode_timestep = step_input['episode_timestep']
        eval_episodes = step_input['eval_episodes']
        scene = step_input['scene']
        self.state = input_action
        self.state = self.state.reshape(1,-1).cpu()
        # print(self.state)
        # self.bf_vec = self.phase2bf(self.state)
        reward, sir, bf_gain, bf_gain_i1, bf_gain_i2 = self.reward_fn(self.state, 0, scene)
        data_rate = 10*math.log10(1+sir)
        terminal = 0
        if episode_timestep==eval_episodes:
            done = True
        else:
            done = False
        return self.state.clone(), reward, bf_gain, bf_gain_i1, bf_gain_i2, sir, data_rate, done, terminal

    def step_sensing(self, step_input):  # input_action: (1, num_ant), rep: phase vector
        input_action = step_input['action']
        episode_timestep = step_input['episode_timestep']
        eval_episodes = step_input['eval_episodes']
        scene = step_input['scene']
        self.state = input_action
        self.state = self.state.reshape(1,-1).cpu()
        # print(self.state)
        # self.bf_vec = self.phase2bf(self.state)
        reward, scr, bf_gain_t, bf_gain_i1, bf_gain_i2 = self.reward_fn_sensing(self.state, 0, scene)
        # inv_Q_pfa = self.inverse_Q_function(1e-2)
        # print(math.sqrt(2*scr))
        detect_prob = self.marcum_q(1,math.sqrt(2*scr),math.sqrt(-2*math.log(1e-6)))
        terminal = 0
        if episode_timestep==eval_episodes:
            done = True
        else:
            done = False
        return self.state.clone(), reward, bf_gain_t, bf_gain_i1, bf_gain_i2, scr, detect_prob, done, terminal

    def reward_fn(self, ph_vec, flag, scene):
        bf_vec = self.phase2bf(ph_vec)
        bf_gain_t, interf_gain_1, interf_gain_2 = self.sir_calc(bf_vec)
        sir = bf_gain_t / (interf_gain_1+interf_gain_2)
        # if bf_gain_t > self.threshold:
        #     reward = np.array([1]).reshape((1, 1))
        #     self.threshold_modif(bf_vec, bf_gain_t, interf_gain_1, interf_gain_2, sir, scene)
        # else:
        #     if bf_gain_t > self.previous_gain:
        #         reward = np.array([0]).reshape((1, 1))
        #     else:
        #         reward = np.array([-1]).reshape((1, 1))
        if sir > self.previous_sir:
            reward = np.array([1]).reshape((1, 1))
            if sir > self.threshold_sir:
                self.threshold_modif(bf_vec, bf_gain_t, interf_gain_1, interf_gain_2, sir, scene)
            # self.previous_sir = sir+0.1
        else:
            reward = np.array([-1]).reshape((1, 1))
        self.previous_sir = sir
        self.previous_gain = bf_gain_t
        return reward, self.threshold_sir, self.bf_gain_t_ach, self.interf_gain_1_ach, self.interf_gain_2_ach

    def get_reward(self, scene, input_action):
        inner_state = input_action

        # Quantization Processing
        # self.options['ph_table_rep'].cuda()
        mat_dist = torch.abs(inner_state.reshape(self.num_ant, 1) - self.options['ph_table_rep'])
        action_quant = self.options['ph_table_rep'][range(self.num_ant), torch.argmin(mat_dist, dim=1)].reshape(1, -1)

        reward, sir, bf_gain_t,_,_ = self.reward_fn(action_quant, 1, scene)
        self.count += 1
        return reward, bf_gain_t, sir, action_quant.clone(), action_quant.clone()
    
    def reward_fn_sensing(self, ph_vec, flag, scene):
        bf_vec = self.phase2bf(ph_vec)
        bf_gain_t, interf_gain_1, interf_gain_2 = self.scr_calc(bf_vec)
        scr = bf_gain_t / (interf_gain_1+interf_gain_2)
        # if bf_gain_t > self.threshold_sensing:
        #     reward = np.array([1]).reshape((1, 1))
        #     self.threshold_sensing_modif(bf_vec, bf_gain_t, interf_gain_1, interf_gain_2, scr, scene)
        # else:
        #     if bf_gain_t > self.previous_sensing_gain:
        #         reward = np.array([0]).reshape((1, 1))
        #     else:
        #         reward = np.array([-1]).reshape((1, 1))
        if scr > self.previous_scr:
            reward = np.array([1]).reshape((1, 1))
            if scr > self.threshold_scr:
                self.threshold_sensing_modif(bf_vec, bf_gain_t, interf_gain_1, interf_gain_2, scr, scene)
            # self.previous_scr = scr+0.1
        else:
            reward = np.array([-1]).reshape((1, 1))
        self.previous_scr = scr
        self.previous_sensing_gain = bf_gain_t
        return reward, self.threshold_scr, self.bf_gain_t_ach_sensing, self.interf_gain_1_ach_sensing, self.interf_gain_2_ach_sensing
    
    def get_reward_sensing(self, scene, input_action):
        inner_state = input_action
        # Quantization Processing
        # self.options['ph_table_rep'].cuda()
        mat_dist = torch.abs(inner_state.reshape(self.num_ant, 1) - self.options['ph_table_rep'])
        action_quant = self.options['ph_table_rep'][range(self.num_ant), torch.argmin(mat_dist, dim=1)].reshape(1, -1)

        reward, scr, bf_gain_t,_,_ = self.reward_fn_sensing(action_quant, 1, scene)
        self.count += 1
        return reward, bf_gain_t, scr, action_quant.clone(), action_quant.clone()

    def threshold_modif(self, bf_vec, bf_gain, int_1, int_2, sir, scene):
        self.bf_gain_t_ach = bf_gain
        self.interf_gain_1_ach = int_1
        self.interf_gain_2_ach = int_2
        self.gain_recording(bf_vec, self.idx, scene)
        # self.threshold += self.th_step
        self.threshold = bf_gain
        self.threshold_sir = sir
    
    def threshold_sensing_modif(self, bf_vec, bf_gain, int_1, int_2, scr, scene):
        self.bf_gain_t_ach_sensing = bf_gain
        self.interf_gain_1_ach_sensing = int_1
        self.interf_gain_2_ach_sensing = int_2
        self.sensing_gain_recording(bf_vec, self.idx, scene)
        # self.threshold += self.th_step
        self.threshold_sensing = bf_gain
        self.threshold_scr = scr

    def opt_bf_gain(self):
        ch_r = torch.Tensor.cpu(self.ch.clone()).numpy()[:, :self.num_ant]
        ch_i = torch.Tensor.cpu(self.ch.clone()).numpy()[:, self.num_ant:]
        # ch_r = torch.Tensor.cpu(self.ch.clone()).numpy()[0, :self.num_ant].reshape((1, -1))
        # ch_i = torch.Tensor.cpu(self.ch.clone()).numpy()[0, self.num_ant:].reshape((1, -1))
        radius = np.sqrt(np.square(ch_r) + np.square(ch_i))
        gain_opt = np.mean(np.square(np.sum(radius, axis=1)))
        print('EGC bf gain: ', gain_opt)
        # return gain_opt

    def phase2bf(self, ph_vec):
        bf_vec = torch.zeros((1, 2 * self.num_ant)).float().cuda()
        # bf_vec = torch.zeros((1, 2 * self.num_ant)).float().cpu()
        # print(ph_vec.size())
        for kk in range(self.num_ant):
            bf_vec[0, 2*kk] = torch.cos(ph_vec[0, kk])
            bf_vec[0, 2*kk+1] = torch.sin(ph_vec[0, kk])
            # bf_vec[0, 2*kk] = torch.cos(ph_vec[kk])
            # bf_vec[0, 2*kk+1] = torch.sin(ph_vec[kk])
        return bf_vec

    def bf_gain_cal(self): # used in self.reward_fn
        bf_r = self.bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = self.bf_vec[0, 1::2].clone().reshape(1, -1)
        ch_r = torch.squeeze(self.ch[:, :self.num_ant].clone())
        ch_i = torch.squeeze(self.ch[:, self.num_ant:].clone())
        bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_r = (bf_gain_1+bf_gain_2)**2
        bf_gain_i = (bf_gain_3-bf_gain_4)**2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain = torch.mean(bf_gain_pattern)
        return bf_gain

    def sir_calc(self, bf_vec):
        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)

        ch_t_r = torch.squeeze(self.ch[0, :self.num_ant].clone())
        ch_t_i = torch.squeeze(self.ch[0, self.num_ant:].clone())
        bf_gain_t_1 = torch.matmul(bf_r, torch.t(ch_t_r))
        bf_gain_t_2 = torch.matmul(bf_i, torch.t(ch_t_i))
        bf_gain_t_3 = torch.matmul(bf_r, torch.t(ch_t_i))
        bf_gain_t_4 = torch.matmul(bf_i, torch.t(ch_t_r))

        bf_gain_t_r = (bf_gain_t_1 + bf_gain_t_2) ** 2
        bf_gain_t_i = (bf_gain_t_3 - bf_gain_t_4) ** 2
        bf_gain_pattern_t = bf_gain_t_r + bf_gain_t_i
        bf_gain_t = torch.mean(bf_gain_pattern_t)  # bf gain of the target user

        ch_i_r = torch.squeeze(self.ch[1, :self.num_ant].clone())
        ch_i_i = torch.squeeze(self.ch[1, self.num_ant:].clone())
        bf_gain_i_1 = torch.matmul(bf_r, torch.t(ch_i_r))
        bf_gain_i_2 = torch.matmul(bf_i, torch.t(ch_i_i))
        bf_gain_i_3 = torch.matmul(bf_r, torch.t(ch_i_i))
        bf_gain_i_4 = torch.matmul(bf_i, torch.t(ch_i_r))

        bf_gain_i_r = (bf_gain_i_1 + bf_gain_i_2) ** 2
        bf_gain_i_i = (bf_gain_i_3 - bf_gain_i_4) ** 2
        bf_gain_pattern_i = bf_gain_i_r + bf_gain_i_i
        interf_gain_1 = torch.mean(bf_gain_pattern_i)  # bf gain of the interf user 1

        ch_i_r = torch.squeeze(self.ch[2, :self.num_ant].clone())
        ch_i_i = torch.squeeze(self.ch[2, self.num_ant:].clone())
        bf_gain_i_1 = torch.matmul(bf_r, torch.t(ch_i_r))
        bf_gain_i_2 = torch.matmul(bf_i, torch.t(ch_i_i))
        bf_gain_i_3 = torch.matmul(bf_r, torch.t(ch_i_i))
        bf_gain_i_4 = torch.matmul(bf_i, torch.t(ch_i_r))

        bf_gain_i_r = (bf_gain_i_1 + bf_gain_i_2) ** 2
        bf_gain_i_i = (bf_gain_i_3 - bf_gain_i_4) ** 2
        bf_gain_pattern_i = bf_gain_i_r + bf_gain_i_i
        interf_gain_2 = torch.mean(bf_gain_pattern_i)  # bf gain of the interf user 2

        return bf_gain_t, interf_gain_1, interf_gain_2

    def scr_calc(self, bf_vec):
        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)

        sv_t_r = self.steering_t_vec[0, ::2].clone().reshape(1, -1)
        sv_t_i = self.steering_t_vec[0, 1::2].clone().reshape(1, -1)
        bf_gain_1 = torch.matmul(bf_r, torch.t(sv_t_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(sv_t_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(sv_t_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(sv_t_r))

        bf_gain_r = (bf_gain_1+bf_gain_2)**2
        bf_gain_i = (bf_gain_3-bf_gain_4)**2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain_t = torch.mean(bf_gain_pattern)

        sv_i1_r = self.steering_i1_vec[0, ::2].clone().reshape(1, -1)
        sv_i1_i = self.steering_i1_vec[0, 1::2].clone().reshape(1, -1)
        bf_gain_1 = torch.matmul(bf_r, torch.t(sv_i1_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(sv_i1_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(sv_i1_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(sv_i1_r))

        bf_gain_r = (bf_gain_1+bf_gain_2)**2
        bf_gain_i = (bf_gain_3-bf_gain_4)**2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain_i1 = torch.mean(bf_gain_pattern)

        sv_i2_r = self.steering_i2_vec[0, ::2].clone().reshape(1, -1)
        sv_i2_i = self.steering_i2_vec[0, 1::2].clone().reshape(1, -1)
        bf_gain_1 = torch.matmul(bf_r, torch.t(sv_i2_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(sv_i2_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(sv_i2_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(sv_i2_r))

        bf_gain_r = (bf_gain_1+bf_gain_2)**2
        bf_gain_i = (bf_gain_3-bf_gain_4)**2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain_i2 = torch.mean(bf_gain_pattern)

        return bf_gain_t, bf_gain_i1, bf_gain_i2
    
    # def bf_gain_sensing_cal(self): # used in self.reward_fn_sensing
    #     bf_r = self.bf_vec[0, ::2].clone().reshape(1, -1)
    #     bf_i = self.bf_vec[0, 1::2].clone().reshape(1, -1)
    #     sv_r = self.steering_vec[0, ::2].clone().reshape(1, -1)
    #     sv_i = self.steering_vec[0, 1::2].clone().reshape(1, -1)
    #     bf_gain_1 = torch.matmul(bf_r, torch.t(sv_r))
    #     bf_gain_2 = torch.matmul(bf_i, torch.t(sv_i))
    #     bf_gain_3 = torch.matmul(bf_r, torch.t(sv_i))
    #     bf_gain_4 = torch.matmul(bf_i, torch.t(sv_r))

    #     bf_gain_r = (bf_gain_1+bf_gain_2)**2
    #     bf_gain_i = (bf_gain_3-bf_gain_4)**2
    #     bf_gain_pattern = bf_gain_r + bf_gain_i
    #     bf_gain = torch.mean(bf_gain_pattern)
    #     return bf_gain

    def bf_gain_cal_only(self, bf_vec): # used in self.get_reward
        bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
        bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)
        ch_r = torch.squeeze(self.ch[:, :self.num_ant].clone())
        ch_i = torch.squeeze(self.ch[:, self.num_ant:].clone())
        bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
        bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
        bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
        bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))

        bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
        bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
        bf_gain_pattern = bf_gain_r + bf_gain_i
        bf_gain = torch.mean(bf_gain_pattern)
        return bf_gain
    
    # def bf_gain_sensing_cal_only(self, bf_vec): # used in self.get_reward_sensing
    #     bf_r = bf_vec[0, ::2].clone().reshape(1, -1)
    #     bf_i = bf_vec[0, 1::2].clone().reshape(1, -1)
    #     sv_r = self.steering_vec[0, ::2].clone().reshape(1, -1)
    #     sv_i = self.steering_vec[0, 1::2].clone().reshape(1, -1)
    #     bf_gain_1 = torch.matmul(bf_r, torch.t(sv_r))
    #     bf_gain_2 = torch.matmul(bf_i, torch.t(sv_i))
    #     bf_gain_3 = torch.matmul(bf_r, torch.t(sv_i))
    #     bf_gain_4 = torch.matmul(bf_i, torch.t(sv_r))

    #     bf_gain_r = (bf_gain_1+bf_gain_2)**2
    #     bf_gain_i = (bf_gain_3-bf_gain_4)**2
    #     bf_gain_pattern = bf_gain_r + bf_gain_i
    #     bf_gain = torch.mean(bf_gain_pattern)
    #     return bf_gain

    def gain_vs_iter(self):
        gain_best = max(self.gain_record).reshape((1, 1))
        if os.path.exists('performance.txt'):
            with open('performance.txt', 'ab') as pf:
                np.savetxt(pf, np.array(self.count).reshape((1, 1)), fmt='%.2f', delimiter='\n')
            with open('performance.txt', 'ab') as pf:
                np.savetxt(pf, gain_best, fmt='%.2f', delimiter='\n')
        else:
            np.savetxt('performance.txt', np.array(self.count).reshape((1, 1)), fmt='%.2f', delimiter='\n')
            with open('performance.txt', 'ab') as pf:
                np.savetxt(pf, gain_best, fmt='%.2f', delimiter='\n')


    def gain_recording(self, bf_vec, idx, scene):
        new_gain = torch.Tensor.cpu(self.bf_gain_t_ach).detach().numpy().reshape((1, 1))
        bf_print = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
        if new_gain > max(self.gain_record):
            self.gain_record.append(new_gain)
            self.best_bf_vec = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
            if os.path.exists('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt'):   #Anik 22-08-23
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')  #Anik 22-08-23
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            else:
                np.savetxt('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', new_gain, fmt='%.2f', delimiter='\n')  #Anik 22-08-23
                # with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                #     np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            self.best_bf_vec = bf_vec

    def sensing_gain_recording(self, bf_vec, idx, scene):
        new_gain = torch.Tensor.cpu(self.bf_gain_t_ach_sensing).detach().numpy().reshape((1, 1))
        bf_print = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
        if new_gain > max(self.sensing_gain_record):
            self.sensing_gain_record.append(new_gain)
            self.best_bf_sensing_vec = torch.Tensor.cpu(bf_vec).detach().numpy().reshape(1, -1)
            if os.path.exists('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt'):   #Anik 22-08-23
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')  #Anik 22-08-23
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            else:
                np.savetxt('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', new_gain, fmt='%.2f', delimiter='\n')  #Anik 22-08-23
                # with open('beams/beams_' + str(idx) + '_max.txt', 'ab') as bm:
                #     np.savetxt(bm, new_gain, fmt='%.2f', delimiter='\n')
                with open('beams_allscenes/beams_'+str(scene)+'/beams_' + str(idx) + '_max.txt', 'ab') as bm:  #Anik 22-08-23
                    np.savetxt(bm, bf_print, fmt='%.5f', delimiter=',')
            self.best_bf_sensing_vec = bf_vec

    def codebook_gen(self):
        angles = np.linspace(0, 2 * np.pi, self.cb_size, endpoint=False)
        cb = np.exp(1j * angles)
        codebook = torch.zeros((self.cb_size, 2)) # shape of the codebook
        for ii in range(cb.shape[0]):
            codebook[ii, 0] = torch.tensor(np.real(cb[ii]))
            codebook[ii, 1] = torch.tensor(np.imag(cb[ii]))
        return codebook

    def steering_vec_gen(self, theta):   #Steering Vec generation
        theta_tensor = torch.zeros((1,self.num_ant))
        sep_tensor = torch.zeros((1,self.num_ant))
        for i in range(self.num_ant):
            theta_tensor[0, i] = torch.tensor(np.sin(theta))
            sep_tensor[0, i] = torch.tensor(i)
        steering_angle = np.exp((2j * np.pi * sep_tensor * self.ant_sep * theta_tensor)/self.wavelength)
        steering_vec = torch.zeros((1, 2*self.num_ant))
        for kk in range(self.num_ant):
            steering_vec[0, 2*kk] = torch.tensor(np.real(steering_angle[0,kk]))
            steering_vec[0, 2*kk+1] = torch.tensor(np.imag(steering_angle[0,kk]))
        return steering_vec

    def init_bf_vec(self):
        bf_vec = torch.empty((1, 2 * self.num_ant))
        bf_vec[0, ::2] = torch.tensor([1])
        bf_vec[0, 1::2] = torch.tensor([0])
        bf_vec = bf_vec.float().cuda()
        # bf_vec = bf_vec.float().cpu()
        return bf_vec

    def init_best(self):
        ph_book = np.linspace(-np.pi, np.pi, 2 ** self.num_bits, endpoint=False)
        ph_vec = np.array([[ph_book[np.random.randint(0, len(ph_book))] for ii in range(self.num_ant)]])
        bf_complex = np.exp(1j * ph_vec)
        bf_vec = np.empty((1, 2 * self.num_ant))
        for kk in range(self.num_ant):
            bf_vec[0, 2*kk] = np.real(bf_complex[0, kk])
            bf_vec[0, 2*kk+1] = np.imag(bf_complex[0, kk])
        return torch.from_numpy(bf_vec).float().cuda()
        # return torch.from_numpy(bf_vec).float().cpu()

    def marcum_q(self,nu,a,b):
        return scipy.stats.ncx2.sf(b**2, 2*nu, a**2)

    def reset(self):
        pass
        