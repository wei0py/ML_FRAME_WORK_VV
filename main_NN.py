#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os
import sys
import parameters as pm
workpath=os.path.abspath(pm.codedir)
sys.path.append(workpath)

# pm.istrain = True
from data_scaler import DataScalers
from nn_model import EiNN
from nn_train import Trainer

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = pm.cuda_dev

#=======================================================================
if not os.path.isdir(pm.dir_work):
    os.system("mkdir " + pm.dir_work)
for dirn in [pm.d_nnEi, pm.d_nnFi]:
    if not os.path.isdir(dirn):
        os.system("mkdir " + dirn)

# import NN_seper
# NN_seper.write_natoms_dfeat()

# import combine_dE
# combine_dE.run_combine_dE()

with tf.device('/device:GPU:0'):

    nn = EiNN()

    if pm.train_continue:
        data_scalers = DataScalers(f_ds=pm.f_data_scaler,
                                   f_feat=pm.f_train_feat, load=True)
        trainer = Trainer(nn, data_scalers)
        ckpt_path = pm.d_nnFi
        latest = tf.train.latest_checkpoint(ckpt_path)
        sess = trainer.init_sess(latest)
        print("Continue from", latest)
    else:
        data_scalers = DataScalers(f_ds=pm.f_data_scaler,
                                   f_feat=pm.f_train_feat)
        trainer = Trainer(nn, data_scalers)
        sess = trainer.init_sess('dummy')

    # if pm.train_stage == 0:
    #     trainer.train_Ei(pm.f_pretr_feat,
    #                      pm.f_test_feat,
    #                      pm.epochs_pretrain, nn_file=pm.d_nnEi+'preEi',
    #                      eMAE_err=pm.eMAE_err, f_err_log=pm.dir_work+'out_err_pretrain.dat')

    #     nn.saveWij_np(sess, pm.d_nnEi+'Wij_pretrain.npy')
        
    if pm.train_stage < 2:
        trainer.train_Ei(pm.f_train_feat,
                            pm.f_test_feat,
                            pm.epochs_alltrain, nn_file=pm.d_nnEi+'allEi', eMAE_err=pm.eMAE_err)

        nn.saveWij_np(sess, pm.d_nnEi+'Wij_train-Ei.npy')

    if pm.train_stage < 3:
        trainer.train_Fi(pm.f_train_feat,
                         pm.f_train_dfeat,
                        #  pm.f_train_nblt,
                         pm.f_train_natoms,
                         pm.f_test_feat,
                         pm.f_test_dfeat,
                        #  pm.f_test_nblt,
                         pm.f_test_natoms,
                         pm.epochs_Fi_train, nn_file=pm.d_nnFi+'Fi', iFi_repeat=pm.iFi_repeat)

        nn.saveWij_np(sess, pm.f_Wij_np)

    if pm.flag_plt:
        plt.show()
    
