#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
import math
import os
import parameters as pm
#TODO:
from prepare import r_feat_csv
from data_scaler import mae, mse

from nn_model import EiNN, NNapiBase
if pm.progressbar:
    import progressbar

from read_dfeatNN import read_dfeatnn

class Trainer(NNapiBase):
    """
    Trainer

    parameters:

    """

    #===========================================================================

    def __init__(self, nn, data_scalers, \
            learning_rate=pm.learning_rate, rtLossE=pm.rtLossE, rtLossF=pm.rtLossF ):

        super(Trainer, self).__init__(nn=nn, data_scalers=data_scalers)

        self.bias_corr = tf.placeholder(tf.bool)

        self.rtLossE = rtLossE # fraction to sum total loss func
        self.rtLossF = rtLossF

        # === self.lossEi and optimizer  ===
        #self.lossEi = tf.reduce_mean(tf.squared_difference(self.pred_Ei, self.YEi))
        self.lossEi = self.get_loss_Ei(self.pred_Ei, self.YEi, self.itp_batch, self.bias_corr)
        self.lossFi = tf.reduce_mean(tf.squared_difference(self.pred_Fi, self.YFi))
        self.lossEF = rtLossE *self.lossEi + rtLossF *self.lossFi
        
        self.optEi  = tf.train.AdamOptimizer(learning_rate).minimize(self.lossEi)
        self.optFi  = tf.train.AdamOptimizer(learning_rate)
        # self.optFi  = tf.train.GradientDescentOptimizer(learning_rate/10.0)
        #self.optFi  = tf.train.AdadeltaOptimizer(learning_rate)
        #self.optFi  = tf.train.RMSPropOptimizer(learning_rate)
        self.optFi_op = self.optFi.minimize(self.lossEF)

    #===========================================================================
    

    def get_loss_Ei(self, pred_Ei, real_Ei, itypes, bias_corr):

        loss_E = 0

        itypes = tf.reshape(itypes, [-1,1])

        alf = tf.zeros([pm.ntypes], dtype=pm.tf_dtype)

        for i in range(self.ntypes):
            idx = tf.equal(itypes[:,0], self.at_types[i])
            alf += tf.cond(tf.equal(bias_corr, tf.constant(True)),
                           lambda: tf.scatter_nd(
                               tf.constant([[i],], dtype=tf.int32),
                               tf.expand_dims(
                               tf.divide(
                                   tf.reduce_sum(tf.squared_difference(
                                       tf.boolean_mask(pred_Ei, idx),
                                       tf.boolean_mask(real_Ei, idx))),
                                   tf.reduce_sum(tf.square(tf.boolean_mask(real_Ei, idx)))
                                   ),-1),
                                   [pm.ntypes]
                               ),
                           lambda: tf.scatter_nd(
                               tf.constant([[i],], dtype=tf.int32),
                               tf.constant([0], dtype=pm.tf_dtype),
                               [pm.ntypes]
                           ))
            #alf = tf.print(alf, [tf.equal(bias_corr, tf.constant(True)),bias_corr, alf])
            loss_E += tf.cond(tf.equal(tf.size(tf.boolean_mask(pred_Ei, idx)), 0),
                              lambda: tf.constant([0], dtype=pm.tf_dtype),
                              lambda: tf.reduce_mean(tf.squared_difference(
                                  tf.boolean_mask(pred_Ei, idx),
                                  (1+alf[i]) * tf.boolean_mask(real_Ei, idx)))
                              )

        return loss_E


    #===========================================================================
        
    def train_Ei(self, f_train, f_test, epochs, # train on energy
                 iprint=100, isaveNN=1000, nn_file=os.path.join(pm.d_nnEi,'pre'), eMAE_err=pm.eMAE_err,
                 f_err_log=pm.dir_work+'/out_err.dat'):
        """
        """
        print("\n=train_Ei start ", f_train, epochs, "epochs, at ", time.ctime())
        itypes,feat,engy = r_feat_csv(f_train)
        feat_scaled = self.ds.pre_feat(feat, itypes)
        engy_scaled = self.ds.pre_engy(engy, itypes)

        print("  and test file ", f_test)
        itypes_t,feat_t,engy_t = r_feat_csv(f_test)

        if pm.flag_plt:
            fig, ax=plt.subplots()
            line_train,=ax.plot([],[], label='train_MAE')
            line_test,=ax.plot([],[], label='test_MAE')
            ax.set_yscale('log')
            ax.legend()
            plt.show(block=False)

        for epoch in range(epochs):
            self.sess.run(self.optEi, 
                          feed_dict={self.X:feat_scaled, self.YEi:engy_scaled, self.itp_batch:itypes, self.bias_corr:pm.bias_corr})
        
            if epoch % iprint == 0 and epoch != 0:
                training_cost, engy_out = self.sess.run(\
                        (self.lossEi, self.pred_Ei), \
                        feed_dict={self.X:feat_scaled, self.YEi:engy_scaled, self.itp_batch:itypes, self.bias_corr:pm.bias_corr})
                print(epoch, training_cost, mae(engy_out,engy_scaled), mse(engy_out,engy_scaled), time.ctime())
                sys.stdout.flush()

                engy_out = self.ds.post_engy(engy_out, itypes)
                
                out_Ep = np.concatenate([np.expand_dims(itypes, -1), engy, engy_out, engy_out -engy], axis=1)
                df_out = pd.DataFrame(out_Ep)
                df_out.to_csv(os.path.join(pm.d_nnEi,'E_pred.csv'), mode='w', header=False, index=False)
                test_MAE = self.getEi_err(itypes_t,feat_t,engy_t,outfile=os.path.join(pm.d_nnEi,'E_pred_test.csv'))
                train_MAE = mae(engy_out,engy)   #TODO:

                if epoch // iprint == 1:
                    fid_err_log = open(f_err_log, 'w')
                else:
                    fid_err_log = open(f_err_log, 'a')
                fid_err_log.write('%d %e %e\n' % (epoch, train_MAE, test_MAE))
                fid_err_log.close()

                if pm.flag_plt:
                    line_train.set_xdata(np.append(line_train.get_xdata(),epoch))
                    line_train.set_ydata(np.append(line_train.get_ydata(),train_MAE))
                    line_test.set_xdata(np.append(line_test.get_xdata(),epoch))
                    line_test.set_ydata(np.append(line_test.get_ydata(),test_MAE))
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                
            if epoch % isaveNN == 0 and epoch != 0:
                save_path = self.saver.save(self.sess, nn_file+str(epoch)+".ckpt")
                print("Model saved: {}".format(save_path))

            if epoch > iprint and (test_MAE < train_MAE*1.5)and(train_MAE < eMAE_err):
                print("=== convergence with test data at ", time.ctime())
                break
        
        save_path = self.saver.save(self.sess, nn_file+"_final.ckpt")
        print("Model saved: {}".format(save_path))
        print("=train_Ei end === at ", time.ctime())
        sys.stdout.flush()
            
    #===========================================================================

    def getEi_err(self, itypes,feat,engy, outfile=os.path.join(pm.d_nnEi,'E_pred_test.csv'), b_print=False):

        feat_scaled = self.ds.pre_feat(feat, itypes)
        engy_scaled = self.ds.pre_engy(engy, itypes)
        
        test_engy = self.sess.run(\
                self.pred_Ei, \
                feed_dict={self.X: feat_scaled, self.YEi: engy_scaled, self.itp_batch:itypes, self.bias_corr:pm.bias_corr})

        test_engy = self.ds.post_engy(test_engy, itypes)

        if outfile != None :
            out_Ep = np.concatenate([np.expand_dims(itypes, -1), engy, test_engy, test_engy -engy], axis=1)
            df_out = pd.DataFrame(out_Ep)
            df_out.to_csv(outfile, mode='w', header=False, index=False)
        
        if b_print == True :
            #print("=getEi_err "+f_test+" =")
            print("=getEi_err_test =")
            print("MAE, MSE: ", mae(test_engy,engy), mse(test_engy,engy))
            sys.stdout.flush()
    
        return mae(test_engy,engy)

    #===========================================================================
    #@profile
    def train_Fi(self, f_train_feat, f_train_dfeat, 
                 f_train_natoms, f_test_feat, f_test_dfeat,  f_test_natoms,
                 epochs, 
                 nn_file=pm.d_nnFi+'Fi', iFi_repeat=20,
                 iprint=10, isaveNN=1, batch_size=pm.batch_size,
                 f_err_log=pm.dir_work+'out_err_for.dat',
                 eMAE_err=pm.eMAE_err, fMAE_err=pm.fMAE_err):

        print("\n=train_Fi start ",  epochs, "epochs, at ", time.ctime())

        if pm.flag_plt:
            figE, axE=plt.subplots()
            figF, axF=plt.subplots()
            line_E_train,=axE.plot([],[], label='train_E_MAE')
            line_E_test,=axE.plot([],[], label='test_E_MAE')
            line_F_train,=axF.plot([],[], label='train_F_MAE')
            line_F_test,=axF.plot([],[], label='test_F_MAE')
            axE.set_yscale('log')
            axE.legend()
            axF.set_yscale('log')
            axF.legend()
            plt.show(block=False)

        natoms = np.loadtxt(f_train_natoms, dtype=np.int); # TODO: natoms contain all atomnum of each image, format: totnatom, type1n, type2 n
        # nImg = natoms.shape[0]
        # indImg = np.zeros((nImg+1,), dtype=np.int)
        # indImg[0] = 0
        # for i in range(nImg):
        #     indImg[i+1] = indImg[i] + natoms[i,0] #TODO: 

        # itypes,feat,engy = r_feat_csv(f_train_feat) #TODO:
        # feat_scaled = self.ds.pre_feat(feat, itypes)
        # engy_scaled = self.ds.pre_engy(engy, itypes)

        dfeat_names = {}
        image_nums = {}
        pos_nums = {}
        for m in pm.use_Ftype:
            dfeat_names[m] = pd.read_csv(f_train_dfeat+str(m), header=None).values[:,0] #TODO:
            image_nums[m] = pd.read_csv(f_train_dfeat+str(m), header=None).values[:,1].astype(int)
            pos_nums[m] = pd.read_csv(f_train_dfeat+str(m), header=None).values[:,2].astype(int)
            nImg = image_nums[m].shape[0]
      
        # fors, nblist = r_fors_nblist_csv(f_train_nblt) #TODO:
        # fors_scaled = self.ds.pre_fors(fors, itypes)
        # dfeat_names = pd.read_csv(f_train_dfeat, header=None).values[:,0] #TODO:

        nbatches = int(math.ceil(nImg/batch_size))
        batch_size_last = int(nImg - batch_size * (nbatches - 1))

        opt = self.optFi
        loss = self.lossEF
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tv.initialized_value(),
                                  trainable=False) for tv in tvs]
        zero_ops  = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = opt.compute_gradients(loss)
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        apply_ops = opt.apply_gradients([(accum_vars[i], tv) for i, tv 
                                         in enumerate(tf.trainable_variables())])
        reset_opt = tf.variables_initializer(opt.variables())
        #print(self.sess.run(tf.report_uninitialized_variables()))
        #quit()
        
        for epoch in range(epochs):
        #for epoch in range(1):

            #self.sess.run(zero_ops)
            
            nextImg = 0 # Batch head
            rndind = np.random.permutation(nImg)

            if pm.progressbar:
                bar = progressbar.progressbar(range(nbatches),
                                              widgets= ['%d '%(epoch), progressbar.Percentage(), ' (', progressbar.SimpleProgress(),
                                                        ') ', progressbar.Bar(marker = '=', left=' [', right='] ', fill='.'),
                                                        progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
                )
            else:
                bar = range(nbatches)
                
            for ibat in bar:
                #self.sess.run(reset_opt)            
                feat_scaled_bat = []
                nblist_bat = []
                engy_scaled_bat = []
                fors_scaled_bat = []
                itypes_bat = []
                batch = []
                if ibat == nbatches - 1:
                    real_batch_size = batch_size_last
                else:
                    real_batch_size = batch_size

                nblt_shift = 0
                for i in range(real_batch_size):
                    # feat_scaled_bat.append(feat_scaled[indImg[rndind[nextImg+i]]:indImg[rndind[nextImg+i]+1]])
                    # # nblist_bat.append(nblist[indImg[rndind[nextImg+i]]:indImg[rndind[nextImg+i]+1]].copy())
                    # engy_scaled_bat.append(engy_scaled[indImg[rndind[nextImg+i]]:indImg[rndind[nextImg+i]+1]])
                    # # fors_scaled_bat.append(fors_scaled[indImg[rndind[nextImg+i]]:indImg[rndind[nextImg+i]+1]])
                    # itypes_bat.append(itypes[indImg[rndind[nextImg+i]]:indImg[rndind[nextImg+i]+1]])
                    kk=0
                    dfeat_name={}
                    image_num={}
                    pos_num={}
                    for mm in pm.use_Ftype:
                        dfeat_name[mm] = dfeat_names[mm][rndind[nextImg+i]]
                        image_num[mm] = image_nums[mm][rndind[nextImg+i]]
                        pos_num[mm] = pos_nums[mm][rndind[nextImg+i]]
                        itype_atom=np.asfortranarray(np.array(pm.atomType).transpose())
                        wp_atom=np.asfortranarray(np.array(pm.fortranFitAtomRepulsingEnergies).transpose())
                        rad_atom=np.asfortranarray(np.array(pm.fortranFitAtomRadii).transpose())
                        read_dfeatnn.read_dfeat(dfeat_name[mm],image_num[mm],pos_num[mm],itype_atom,rad_atom,wp_atom)

                        feat_tmp=np.array(read_dfeatnn.feat).transpose().astype(pm.tf_dtype)

                        dfeat_tmp=np.array(read_dfeatnn.dfeat).transpose(1,2,0,3).astype(pm.tf_dtype)
                        if kk==0:
                            feat=feat_tmp
                            dfeat=dfeat_tmp
                            fors = np.array(read_dfeatnn.force).transpose().astype(pm.tf_dtype)
                            nblist = np.array(read_dfeatnn.list_neigh).transpose().astype(int)
                            engy=np.array(read_dfeatnn.energy).reshape((-1,1)).astype(pm.tf_dtype)
                            itypes=np.array(read_dfeatnn.iatom).transpose().astype(int)
                        else:
                            feat=np.concatenate((feat,feat_tmp),axis=1)
                            dfeat=np.concatenate((dfeat,dfeat_tmp),axis=2)
                        read_dfeatnn.deallo()
                        kk=kk+1
                    feat_scaled_bat.append(self.ds.pre_feat(feat,itypes))
                    engy_scaled_bat.append(self.ds.pre_engy(engy, itypes))
                    itypes_bat.append(itypes)
                    fors_scaled = self.ds.pre_fors(fors, itypes)
                    nblist_bat.append(nblist)
                    fors_scaled_bat.append(fors_scaled)
                    batch.append(self.ds.pre_dfeat(dfeat, itypes_bat[-1], nblist_bat[-1]))

                    nblist_bat[-1][nblist_bat[-1]>0] += nblt_shift
                    #nblist_bat[i] += nblt_shift
                    nblt_shift += natoms[rndind[nextImg+i], 0]
                    

                # read_dfeatnn.deallo()

                dfeat_scaled = np.concatenate(batch, axis = 0)
                feat_scaled_bat = np.concatenate(feat_scaled_bat, axis = 0)
                nblist_bat = np.concatenate(nblist_bat, axis = 0)
                engy_scaled_bat = np.concatenate(engy_scaled_bat, axis = 0)
                fors_scaled_bat = np.concatenate(fors_scaled_bat, axis = 0)
                itypes_bat = np.concatenate(itypes_bat, axis = 0)

    

                nextImg += real_batch_size


                for i in range(iFi_repeat):
                    lossEF, lossF_,loss_, f_out, engy_out = \
                        self.sess.run( 
                            (self.optFi_op, self.lossFi, self.lossEi, self.pred_Fi, self.pred_Ei),
                        #accum_ops, \
                            feed_dict={self.X:feat_scaled_bat, \
                                       self.tfdXin:dfeat_scaled, self.tf_idxNb:nblist_bat,\
                                       self.YEi:engy_scaled_bat, self.YFi:fors_scaled_bat,\
                                       self.itp_batch:itypes_bat, self.natoms_batch:itypes_bat.shape[0],\
                                       self.bias_corr:pm.bias_corr})

                    #'''
                    if pm.train_verb > 0:
                        sys.stdout.flush()
                        print(ibat, i, "loss_F: ", lossF_, \
                              "\tloss_E: ", loss_ , \
                              " losstot:", self.rtLossE *loss_ + self.rtLossF *lossF_, \
                              "at ", time.ctime())
                        print(ibat, i, "Fi_MSE: ", mse(f_out,fors_scaled_bat), \
                              "\tEi_MSE: ", mse(engy_out,engy_scaled_bat), \
                              " dEi_sum", np.sum(engy_out-engy_scaled_bat) )
                        sys.stdout.flush()
                    #'''

            #self.sess.run(apply_ops)
                    
            if epoch % iprint == 0 and epoch != 0:
                print("=== ", int(epoch/iprint), "th check error at ", time.ctime())
                #self.getEi_err(pm.f_test_feat)
                train_E_MAE, train_E_MSE, train_F_MAE, train_F_MSE =\
                    self.getFi_err(f_train_feat, f_train_dfeat, f_train_natoms, outfile=pm.dir_work+'out_train_ckFi')
                test_E_MAE, test_E_MSE, test_F_MAE, test_F_MSE = \
                    self.getFi_err(f_test_feat, f_test_dfeat, f_test_natoms, outfile=pm.dir_work+'out_test_ckFi')
                print(epoch, train_E_MAE, test_E_MAE, train_E_MSE, test_E_MSE,
                      train_F_MAE, test_F_MAE, train_F_MSE, test_F_MSE,
                      time.ctime())
                if (epoch // iprint == 1):
                    fid_err_log = open(f_err_log, 'w')
                else:
                    fid_err_log = open(f_err_log, 'a')
                fid_err_log.write('%d %e %e %e %e %e %e %e %e\n' \
                                  % (epoch, train_E_MAE, test_E_MAE, train_E_MSE, test_E_MSE,\
                                     train_F_MAE, test_F_MAE, train_F_MSE, test_F_MSE))
                fid_err_log.close()

                if pm.flag_plt:
                    line_E_train.set_xdata(np.append(line_E_train.get_xdata(),epoch))
                    line_E_train.set_ydata(np.append(line_E_train.get_ydata(),train_E_MAE))
                    line_E_test.set_xdata(np.append(line_E_test.get_xdata(),epoch))
                    line_E_test.set_ydata(np.append(line_E_test.get_ydata(),test_E_MAE))
                    line_F_train.set_xdata(np.append(line_F_train.get_xdata(),epoch))
                    line_F_train.set_ydata(np.append(line_F_train.get_ydata(),train_F_MAE))
                    line_F_test.set_xdata(np.append(line_F_test.get_xdata(),epoch))
                    line_F_test.set_ydata(np.append(line_F_test.get_ydata(),test_F_MAE))
                    axE.relim()
                    axE.autoscale_view()
                    figE.canvas.draw()
                    figE.canvas.flush_events()
                    axF.relim()
                    axF.autoscale_view()
                    figF.canvas.draw()
                    figF.canvas.flush_events()

            if epoch % isaveNN == 0 and epoch != 0:
                save_path = self.saver.save(self.sess, nn_file+str(epoch)+".ckpt")
                print("Model saved: {}".format(save_path))
                sys.stdout.flush()

            if epoch > iprint and ( test_E_MAE < train_E_MAE*1.5) & ( train_E_MAE < eMAE_err ) & ( test_F_MAE < train_F_MAE*1.5)&( train_F_MAE < fMAE_err ):
                print("=== convergence with test data at ", time.ctime())
                break

        save_path = self.saver.save(self.sess, nn_file+"_final.ckpt")
        print("Model saved: {}".format(save_path))
        print("=train_Fi end === at ", time.ctime())
        sys.stdout.flush()

    #=======================================================================

    def getFi_err(self, f_test_feat, f_test_dfeat, f_test_natoms, outfile=None,
                  b_print=False, batch_size=pm.batch_size):

        natoms = np.loadtxt(f_test_natoms, dtype=np.int);
        # nImg = natoms.shape[0]
        # indImg = np.zeros((nImg+1,), dtype=np.int)
        # indImg[0] = 0
        # for i in range(nImg):
        #     indImg[i+1] = indImg[i] + natoms[i,0]
        nextImg = 0

        # itypes,feat,engy = r_feat_csv(f_test_feat)
        # feat_scaled = self.ds.pre_feat(feat, itypes)
        # engy_scaled = self.ds.pre_engy(engy, itypes)
        # fors, nblist = r_fors_nblist_csv(f_test_nblt)
        # fors_scaled = self.ds.pre_fors(fors, itypes)
        dfeat_names = {}
        image_nums = {}
        pos_nums = {}
        for m in pm.use_Ftype:
            dfeat_names[m] = pd.read_csv(f_test_dfeat+str(m), header=None).values[:,0]
            image_nums[m] = pd.read_csv(f_test_dfeat+str(m), header=None).values[:,1].astype(int)
            pos_nums[m] = pd.read_csv(f_test_dfeat+str(m), header=None).values[:,2].astype(int)
            nImg = image_nums[m].shape[0]
        

        nbatches = int(math.ceil(nImg/batch_size))
        batch_size_last = int(nImg - batch_size * (nbatches - 1))

        FiAE=0.
        EiAE=0.
        FiSE = 0.
        EiSE = 0.
        natomstot=0
        if pm.progressbar:
            bar = progressbar.progressbar(range(nbatches),
        #for ibat in progressbar.progressbar(range(3),
                    widgets= [progressbar.Percentage(), ' (', progressbar.SimpleProgress(),
                              ') ', progressbar.Bar(marker = '=', left=' [', right='] ', fill='.'),
                              progressbar.Timer(), ' ', progressbar.AdaptiveETA()]
        )
        else:
            bar = range(nbatches);
            
        for ibat in bar:
            feat_scaled_bat = []
            nblist_bat = []
            engy_scaled_bat = []
            engy_bat = []
            fors_scaled_bat = []
            fors_bat = []
            itypes_bat = []
            batch = []
            if ibat == nbatches - 1:
                real_batch_size = batch_size_last
            else:
                real_batch_size = batch_size

            nblt_shift = 0
            for i in range(real_batch_size): 
                kk=0
                dfeat_name={}
                image_num={}
                pos_num={}
                for mm in pm.use_Ftype:
                    dfeat_name[mm] = dfeat_names[mm][nextImg+i]
                    image_num[mm] = image_nums[mm][nextImg+i]
                    pos_num[mm] = pos_nums[mm][nextImg+i]
                    itype_atom=np.asfortranarray(np.array(pm.atomType).transpose())
                    wp_atom=np.asfortranarray(np.array(pm.fortranFitAtomRepulsingEnergies).transpose())
                    rad_atom=np.asfortranarray(np.array(pm.fortranFitAtomRadii).transpose())
                    read_dfeatnn.read_dfeat(dfeat_name[mm],image_num[mm],pos_num[mm],itype_atom,rad_atom,wp_atom)

                    feat_tmp=np.array(read_dfeatnn.feat).transpose().astype(pm.tf_dtype)
                    dfeat_tmp=np.array(read_dfeatnn.dfeat).transpose(1,2,0,3).astype(pm.tf_dtype)
                    if kk==0:
                        feat=feat_tmp
                        dfeat=dfeat_tmp
                        fors = np.array(read_dfeatnn.force).transpose().astype(pm.tf_dtype)
                        nblist = np.array(read_dfeatnn.list_neigh).transpose().astype(int)
                        engy=np.array(read_dfeatnn.energy).reshape((-1,1)).astype(pm.tf_dtype)
                        itypes=np.array(read_dfeatnn.iatom).transpose().astype(int)
                    else:
                        feat=np.concatenate((feat,feat_tmp),axis=1)
                        dfeat=np.concatenate((dfeat,dfeat_tmp),axis=2)
                    read_dfeatnn.deallo()
                    kk=kk+1
 
                feat_scaled_bat.append(self.ds.pre_feat(feat,itypes))
                engy_scaled_bat.append(self.ds.pre_engy(engy, itypes))
                engy_bat.append(engy)
                itypes_bat.append(itypes)

                fors_scaled = self.ds.pre_fors(fors, itypes)
                # fors_scaled = self.ds.pre_fors(fors, itypes[indImg[nextImg+i]:indImg[nextImg+i+1]])
                nblist_bat.append(nblist)
                fors_scaled_bat.append(fors_scaled)
                fors_bat.append(fors)
                batch.append(self.ds.pre_dfeat(dfeat, itypes_bat[-1], nblist_bat[-1]))

                
                nblist_bat[-1][nblist_bat[-1]>0] += nblt_shift
                nblt_shift += natoms[nextImg+i, 0]
                # read_dfeatnn.deallo()

            dfeat_scaled = np.concatenate(batch, axis = 0)
            feat_scaled_bat = np.concatenate(feat_scaled_bat, axis = 0)
            nblist_bat = np.concatenate(nblist_bat, axis = 0)
            engy_scaled_bat = np.concatenate(engy_scaled_bat, axis = 0)
            engy_bat = np.concatenate(engy_bat, axis = 0)
            fors_scaled_bat = np.concatenate(fors_scaled_bat, axis = 0)
            fors_bat = np.concatenate(fors_bat, axis = 0)
            itypes_bat = np.concatenate(itypes_bat, axis = 0)

  

            nextImg += real_batch_size
            
            engy_out, f_out = \
                    self.sess.run( \
                    (self.pred_Ei, self.pred_Fi), \
                    feed_dict={self.X:feat_scaled_bat, \
                            self.tfdXin:dfeat_scaled, self.tf_idxNb:nblist_bat,\
                            self.YEi:engy_scaled_bat, self.YFi:fors_scaled_bat, \
                               self.itp_batch:itypes_bat, self.natoms_batch:itypes_bat.shape[0]})

            engy_out = self.ds.post_engy(engy_out, itypes_bat)
            f_out = self.ds.post_fors(f_out, itypes_bat)

            FiAE0 = np.sum(np.sqrt(np.sum((f_out-fors_bat)**2, axis = 1)))
            FiAE += FiAE0
            EiAE0 = np.sum(np.abs(engy_out-engy_bat))
            EiAE += EiAE0
            
            FiSE0 = np.sum((f_out-fors_bat)**2)
            FiSE += FiSE0
            EiSE0 = np.sum((engy_out-engy_bat)**2)
            EiSE += EiSE0

            natomstot=natomstot+itypes_bat.shape[0]
    
            if outfile != None :
                out_Fp = np.concatenate([np.expand_dims(itypes_bat,-1), engy_bat, engy_out, engy_bat-engy_out, fors_bat, f_out, fors_bat-f_out], axis=1)
                df_out = pd.DataFrame(out_Fp)
                if ibat==0:
                    df_out.to_csv(outfile, mode='w', header=False, index=False)
                else:
                    df_out.to_csv(outfile, mode='a', header=False, index=False)
                    
        sys.stdout.flush()

        # natomstot=itypes_bat.shape[0]

        return EiAE/natomstot, math.sqrt(EiSE/natomstot), FiAE/natomstot, math.sqrt(FiSE/natomstot/3)
