#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

import parameters as pm

if pm.tf_dtype == 'float32' :
    tf_dtype = tf.float32
    print('info: tf.dtype = tf.float32 in Tensorflow trainning.')
else:
    tf_dtype = tf.float64
    print('info: tf.dtype = tf.float64 in Tensorflow trainning, it may be slower.')
import prepare as pp
pp.readFeatnum()
#===============================================================================
class EiNN:
    """
    Ei Neural Network (EiNN)

    parameters:

    """

    def __init__(self, 
                 nlayers=pm.nLayers, # number of layers, doesn't include the input, but include the final output
                 nodes_In=pm.nFeats, nnodes=pm.nNodes, # nNodes: number of nodes, for each layer, element pair
                 maxNb=pm.maxNeighborNum, f_atoms=pm.f_atoms, linear_ratio = 0.0):

        self.nIn = nodes_In # number of features (np.array)
        self.nInMax = nodes_In.max()
        self.maxNb  = maxNb # max number of neighbors
        self.nlayers = nlayers
        self.nnodes = nnodes

        #self.act = self.dlin
        #self.int_act = self.lin
        self.act = self.elup1
        self.int_act = self.int_elup1
        #self.act = self.d_sigmoid
        #self.int_act = self.sigmoid


        self.at_types = pm.atomType
        self.ntypes = pm.ntypes
        self.b_init=pm.b_init

        for itp in range(self.ntypes): # each element has its own network, initiating W and Bs
            with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
                W=[]
                B=[]
                W.append(tf.get_variable("weights_L0_itp"+str(itp), shape=[self.nIn[itp],self.nnodes[0, itp]],
                                         dtype=tf_dtype, initializer=tf.contrib.layers.xavier_initializer()))
                B.append(tf.get_variable("biases_L0_itp"+str(itp), shape=[1,self.nnodes[0, itp]],
                                         dtype=tf_dtype, initializer=tf.contrib.layers.xavier_initializer()))
                for ilayer in range(1, self.nlayers-1):
                    W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), shape=[self.nnodes[ilayer-1, itp],self.nnodes[ilayer, itp]],
                                             dtype=tf_dtype, initializer=tf.contrib.layers.xavier_initializer()))
                    B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp), shape=[1 ,self.nnodes[ilayer, itp]],
                                             dtype=tf_dtype, initializer=tf.contrib.layers.xavier_initializer()))
                ilayer=self.nlayers-1
                W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), shape=[self.nnodes[ilayer-1, itp],self.nnodes[ilayer, itp]],
                                            dtype=tf_dtype, initializer=tf.contrib.layers.xavier_initializer()))
                B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp), shape=[1 ,self.nnodes[ilayer, itp]],
                                            dtype=tf_dtype, initializer=tf.constant_initializer(self.b_init[itp])))
               
    def _preprocessor(self, features):
        return

    def int_elup1(self, tensor_in): # activation function, integration of eLu(x)+1
        idx = tf.greater_equal(tensor_in, 0.)
        nidx = tf.logical_not(idx)
        right = 0.5 * tf.square(tf.boolean_mask(tensor_in, idx)) + tf.boolean_mask(tensor_in, idx)
        left = tf.exp(tf.boolean_mask(tensor_in,nidx)) - 1
        tensor_out = tf.zeros_like(tensor_in)
        tensor_out += tf.scatter_nd(tf.where(idx), right, tf.shape(tensor_out, out_type = tf.int64))
        tensor_out += tf.scatter_nd(tf.where(nidx), left, tf.shape(tensor_out, out_type = tf.int64))
        return tensor_out

    def elup1(self, tensor_in): # activation function, eLu(x)+1
        return(tf.nn.elu(tensor_in)+1)

    def sigmoid(self, tensor_in):
        return(tf.nn.sigmoid(tensor_in))

    def d_sigmoid(self, tensor_in):
        return(tf.nn.sigmoid(tensor_in)*(1-tf.nn.sigmoid(tensor_in)))

    def lin(self, tensor_in): # activation function, linear
        return(tensor_in)

    def dlin(self, tensor_in): # activation function, derivative of linear
        return(tf.ones_like(tensor_in))

    def getEi_itp(self, itp, features): # simple output. The last layers doesn't have activation
        with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
            W = []
            B = []
            L = []

            W.append(tf.get_variable("weights_L0_itp"+str(itp), dtype=tf_dtype))
            B.append(tf.get_variable("biases_L0_itp"+str(itp), dtype=tf_dtype))
            #L.append(tf.nn.softplus(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            L.append(self.int_act(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            #L.append(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0])

            for ilayer in range(1, self.nlayers-1):
                W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                         dtype=tf_dtype))
                B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                         dtype=tf_dtype))
                #L.append(tf.nn.softplus(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))
                L.append(self.int_act(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))

            ilayer += 1
            W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                     dtype=tf_dtype))
            B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                     dtype=tf_dtype))
            L.append(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer])
            
            #print("NN.getEi_itp, L3out", L3out)

        return L[ilayer]


    def getEi(self, features, itypes): # As features are read in batch, this function calculate the energies element by element, and then put them together
        """
        """
        #print('\n\n getEi()===')
        #print("itypes", itypes)
        itypes = tf.expand_dims(itypes, -1)
        #print("itypes", itypes)

        Ei = tf.zeros_like(itypes, dtype=tf_dtype)
        #print('Ei', Ei)
        for i in range(self.ntypes):
            idx = tf.equal(itypes[:,0], self.at_types[i])
            #print('idx', idx)
            iEi = self.getEi_itp(i, tf.boolean_mask(features, idx))
            #print('iEi', iEi)
            Ei += tf.scatter_nd(tf.where(idx), iEi, tf.shape(Ei,out_type=tf.int64))

        return Ei


    def tf_get_dEldXi(self, itp, features): #dE/dXi, X is a feature
        with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
            W = []
            B = []
            L = []
            dL = []

            W.append(tf.get_variable("weights_L0_itp"+str(itp), dtype=tf_dtype))
            B.append(tf.get_variable("biases_L0_itp"+str(itp), dtype=tf_dtype))
            #dL.append(tf.nn.sigmoid(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            #L.append(tf.nn.softplus(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            dL.append(self.act(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            L.append(self.int_act(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
            #L.append(tf.matmul(features[:,:self.nIn[itp]], W[0])+B[0])
            #dL.append(tf.ones_like(L[0]))

            for ilayer in range(1, self.nlayers-1):
                W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                         dtype=tf_dtype))
                B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                         dtype=tf_dtype))
                #dL.append(tf.nn.sigmoid(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))
                #L.append(tf.nn.softplus(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))
                dL.append(self.act(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))
                L.append(self.int_act(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer]))

            ilayer += 1
            W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                     dtype=tf_dtype))
            B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                     dtype=tf_dtype))
            #L.append(tf.matmul(L[ilayer-1], W[ilayer])+B[ilayer])


        w_j = tf.transpose(W[ilayer])
        ilayer -= 1

        # while ilayer >= 0:
        #     w_j = tf.expand_dims(dL[ilayer]*w_j,1) * tf.expand_dims(W[ilayer],0)
        #     w_j = tf.reduce_sum(w_j, axis=2)
        #     ilayer -= 1

        while ilayer >= 0:
            w_j = tf.expand_dims(dL[ilayer]*w_j,1) * tf.expand_dims(W[ilayer],0)
            w_j = tf.reduce_sum(w_j, axis=2)
            ilayer -= 1

        dEldXi = w_j
        # tf_dEldXi=dEldXi
        o_zeros = tf.zeros((tf.shape(features)[0], self.nInMax-self.nIn[itp]), dtype=tf_dtype)
        tf_dEldXi = tf.concat([dEldXi,o_zeros], axis=1) # padding features to the maximum number, currently useless
        #print("NN.tf_get_dEldXi, tf_dEldXi", tf_dEldXi)
    
        return tf_dEldXi

    def getFi(self, features, dfeatures, tf_idxNb, itp_batch, natoms_batch):

        """
        defult nImg=1
        """
        #print('\n\n getFi()===')

        tf_dEldXi = tf.zeros_like(features, dtype=tf_dtype)
        #print('tf_dEldXi', tf_dEldXi)
        for i in range(self.ntypes):
            idx = tf.equal(itp_batch, self.at_types[i])
            #print('idx', idx)
            i_dEldXi = self.tf_get_dEldXi(i, tf.boolean_mask(features, idx))
            #print('i_dEldXi', i_dEldXi)
            tf_dEldXi += tf.scatter_nd(tf.where(idx), i_dEldXi, tf.shape(tf_dEldXi,out_type=tf.int64))
        
        #TODO:
        # Fll = tf.reduce_sum(tf.expand_dims(tf_dEldXi,2)*dfeatures,axis=[1,2])
        # print('Fll:',Fll.shape)
        #TODO:
        # dENldXi  = tf.gather_nd(tf_dEldXi,
        #                         tf.expand_dims(
        #                             tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1,
        #                             -1)
        #                         )

        dENldXi  = tf.gather_nd(tf_dEldXi,tf.expand_dims(tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1,-1))
        # dENldXi  = tf.gather_nd(tf_dEldXi,
        #                         tf.expand_dims(tf.transpose(tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1),1)
        #                         )
                                
        #dENldXi = tf.print(dENldXi, [tf.expand_dims(tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1,-1)],summarize=500000)
        #tf_idxNb = tf.print(tf_idxNb, [tf.shape(tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1)], summarize=500000)
        #tf_idxNb = tf.print(tf_idxNb, [tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1, tf.shape(tf.boolean_mask(tf_idxNb, tf.greater(tf_idxNb,0))-1),tf.where(tf.greater(tf_idxNb,0))], summarize=500000)
        dEnldXin = tf.scatter_nd(tf.where(tf.greater(tf_idxNb,0)),dENldXi, (natoms_batch, self.maxNb, self.nInMax))
            
        #print("getFi().dENldXi \n",
        #       dENldXi )
        #print("getFi().dEnldXin \n",
        #        dEnldXin )

        Fln = tf.reduce_sum(tf.expand_dims(dEnldXin,3)*dfeatures,axis=[1,2])

        # print('Fln:',Fln.shape)
        #TODO:
        # F_pred=Fln+Fll
        # Fln: feature gradient of the atom as end atoms
        return  Fln

    def saveWij_np(self, sess, f_npfile):
        nnWij = []
        for itp in range(self.ntypes):
            with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
                W = []
                B = []
                for ilayer in range(self.nlayers):
                    W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                             dtype=tf_dtype))
                    B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                             dtype=tf_dtype))
                    nnWij.append(np.array(W[ilayer].eval(session=sess)))
                    nnWij.append(np.array(B[ilayer].eval(session=sess)))
                    #different from Miao Ling's code, this version writes each layer first, as is natural. The next function is revised accordingly

        nnWij = np.array(nnWij)
        np.save(f_npfile, nnWij)
        print('EiNN.saveWij_np to', f_npfile, nnWij.dtype, nnWij.shape)
        return

    def loadWij_np_check(self, sess, f_npfile):
        nnWij = np.load(f_npfile)
        print('EiNN.loadWij_np_check from', f_npfile, nnWij.dtype, nnWij.shape)

        err = 0
        for itp in range(self.ntypes):
            with tf.variable_scope('NN', reuse=tf.AUTO_REUSE):
                W = []
                B = []
                for ilayer in range(self.nlayers):
                    W.append(tf.get_variable("weights_L"+str(ilayer)+"_itp"+str(itp), 
                                             dtype=tf_dtype))
                    err += np.sum(nnWij[itp*2*self.nlayers+ilayer*2+0] -np.array(W[ilayer].eval(session=sess)))
                    B.append(tf.get_variable("biases_L"+str(ilayer)+"_itp"+str(itp),
                                             dtype=tf_dtype))
                    err += np.sum(nnWij[itp*2*self.nlayers+ilayer*2+1] -np.array(B[ilayer].eval(session=sess)))

                    sess.run(W[ilayer].assign(nnWij[itp*2*self.nlayers+ilayer*2+0]))
                    sess.run(B[ilayer].assign(nnWij[itp*2*self.nlayers+ilayer*2+1]))

        print('sum(diff_Wij_Bi) =', err)
        return 


#===============================================================================
class NNapiBase:
    """
    NNapiBase

    parameters:

    """

    #===========================================================================
    def __init__(self, nn, data_scalers):

        self.maxNb  = nn.maxNb
        self.ntypes = nn.ntypes
        self.at_types = nn.at_types

        # === input layer ===
        self.X   = tf.placeholder(tf_dtype, shape=(None, nn.nInMax),name="X") # input (features)
        self.YEi = tf.placeholder(tf_dtype, shape=(None,1),name="YEi") # energy real value

        self.tfdXi    = tf.placeholder(tf_dtype, shape=(None, nn.nInMax,3)) # dE/dX as center atom
        self.tfdXin   = tf.placeholder(tf_dtype, shape=(None, nn.maxNb,nn.nInMax,3)) # dE/dX as end atom
        self.tf_idxNb = tf.placeholder(tf.int64, shape=(None, nn.maxNb)) # indecies of neighbors
        self.YFi = tf.placeholder(tf_dtype, shape=(None,3),name="YFi") # Force real value

        self.itp_batch = tf.placeholder(tf.int64, shape=[None])
        self.natoms_batch = tf.placeholder(tf.int64)

        # === output layer ===
        self.pred_Ei = nn.getEi(self.X, self.itp_batch) # energy output
        self.pred_Fi = nn.getFi(self.X, self.tfdXin, self.tf_idxNb, self.itp_batch, self.natoms_batch) # force output

        # === data scaler process  ===
        self.ds = data_scalers

    #===========================================================================
    def init_sess(self, nn_file):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = pm.gpu_mem
        self.sess = tf.Session(config=config)

        self.saver = tf.train.Saver()
        print("\ninit_sess(), NN expect to be restored from ", nn_file)
        if os.path.exists(nn_file +".index" ):
            self.saver.restore(self.sess, nn_file)
            print("NN is restored succesfully") 
        else:
            print("Warning! There is NO NN_model file ", nn_file)
            print("Will init a new NN ...")
            self.sess.run(tf.global_variables_initializer())

        return self.sess

    def get_sess(self):
        return self.sess

