#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import parameters as pm

import os

if pm.cupyFeat:
    import cupy as cp
else:
    import numpy as cp
import numpy as np

import parameters as pm
# from funcs.atoms_struct import get_atoms_type
import prepare as pp
pp.readFeatnum()

class EiNN_cupy:
    """
    Ei Neural Network (EiNN), cupy version

    parameters:

    """

    def __init__(self, 
                 nlayers=pm.nLayers, # number of layers, doesn't include the input, but include the final output
                 nodes_In=pm.nFeats, nnodes=pm.nNodes, # nNodes: number of nodes, for each layer, element pair
                 maxNb=pm.maxNeighborNum,
                 f_Wij_np=pm.f_Wij_np, f_atoms=pm.f_atoms):

        self.nIn = nodes_In
        self.nInMax = nodes_In.max()
        self.maxNb  = maxNb
        self.nlayers = nlayers
        self.nnodes = nnodes
        self.at_types = cp.reshape(cp.asarray(pm.atomType), [-1,1])
        # self.itp_uc, self.at_types, _ = get_atoms_type(f_atoms)
        self.natoms = 0
        self.ntypes = self.at_types.shape[0]

        if pm.activation_func=='elup1':
            self.act = self.elup1
            self.int_act = self.int_elup1
        # if pm.activation_func=='sigmoid':
        #     self.act = self.d_sigmoid
        #     self.int_act = self.sigmoid
        if pm.activation_func=='softplus':
            self.act = self.sigmoid
            self.int_act = self.softplus
        # self.act = self.elup1
        # self.int_act = self.int_elup1

        self.nnWij = self.loadWij_np(f_Wij_np, b_print=True)

    def _preprocessor(self, features):
        return 

    def sigmoid(self, x):
        return (cp.where(x>= -150, 1.0 /(1.0 +cp.exp(-x)), 0.0))

    def softplus(self, x):
        return (cp.where(x <= 150, cp.log(cp.exp(x)+1), x))

    def elup1(self, x):
        return(cp.where(x >= 0, x + 1, cp.exp(x)))

    def int_elup1(self, x):
        return(cp.where(x >= 0, 0.5 * cp.square(x) + x, cp.exp(x) - 1))

    def getEi_itp(self, itp, features):
        W = []
        B = []
        L = []
        for ilayer in range(self.nlayers):
            W.append(cp.asarray(self.nnWij[itp*2*self.nlayers+ilayer*2+0]))
            B.append(cp.asarray(self.nnWij[itp*2*self.nlayers+ilayer*2+1]))

        L.append(self.int_act(cp.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
        #L.append(cp.matmul(features[:,:self.nIn[itp]], W[0])+B[0])
        for ilayer in range(1, self.nlayers - 1):
            L.append(self.int_act(cp.matmul(L[ilayer - 1], W[ilayer])+B[ilayer]))
        ilayer += 1
        L.append(cp.matmul(L[ilayer - 1], W[ilayer])+B[ilayer])

        return cp.asarray(L[ilayer])

    def getEi(self, features, itypes):
        """
        """
        itypes = cp.reshape(cp.asarray(itypes), [-1,1])
        features=cp.asarray(features)
        # print(features.shape)

        Ei = cp.zeros_like(itypes, dtype=cp.float)
        for i in range(self.ntypes):
            idx = (itypes[:,0] == self.at_types[i, 0])
            # print(idx)
            # print(i)
            # print(features[idx])
            # print(self.getEi_itp(i, features[idx]))
            Ei[idx] = self.getEi_itp(i, features[idx])

        return Ei
    
    def cp_get_dEldXi(self, itp, features):
        W = []
        B = []
        L = []
        dL = []
        for ilayer in range(self.nlayers):
            W.append(cp.asarray(self.nnWij[itp*2*self.nlayers+ilayer*2+0]))
            B.append(cp.asarray(self.nnWij[itp*2*self.nlayers+ilayer*2+1]))

        dL.append(self.act(cp.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
        L.append(self.int_act(cp.matmul(features[:,:self.nIn[itp]], W[0])+B[0]))
        #L.append(cp.matmul(features[:,:self.nIn[itp]], W[0])+B[0])
        #dL.append(cp.ones_like(L[0]))
        for ilayer in range(1, self.nlayers - 1):
            dL.append(self.act(cp.matmul(L[ilayer - 1], W[ilayer])+B[ilayer]))
            L.append(self.int_act(cp.matmul(L[ilayer - 1], W[ilayer])+B[ilayer]))
        ilayer += 1
        #L.append(cp.matmul(L[ilayer - 1], W[ialyer])+B[ilayer])

        w_j = cp.transpose(W[ilayer])
        ilayer -= 1

        while ilayer >= 0:
            w_j = cp.expand_dims(dL[ilayer]*w_j,1) * cp.expand_dims(W[ilayer],0)
            w_j = cp.sum(w_j, axis=2)
            ilayer -= 1
            
        dEldXi = w_j

        #o_zeros = cp.zeros((self.at_types[itp,1], self.nInMax -self.nIn[itp]))
        o_zeros = cp.zeros((features.shape[0], self.nInMax -self.nIn[itp]))
        cp_dEldXi = cp.concatenate([dEldXi,o_zeros], axis=1)
    
        return cp_dEldXi

    def getFi(self, features, dfeatures, cp_idxNb, itp_uc):
        """
        defult nImg=1
        """
        itp_uc=cp.reshape(cp.asarray(itp_uc), [-1,1])
        # cp_idxNb=cp.asarray(cp_idxNb)
        features=cp.asarray(features)
        dfeatures=cp.asarray(dfeatures)
        cp_dEldXi = cp.zeros_like(features, dtype=cp.float)
        self.natoms=features.shape[0]
        

        for i in range(self.ntypes):
            idx = (itp_uc[:,0] == self.at_types[i, 0])
            cp_dEldXi[idx] = self.cp_get_dEldXi(i, features[idx])

        dEnldXin = cp.zeros((self.natoms, self.maxNb, self.nInMax))

        dEnldXin[cp_idxNb>0] = cp_dEldXi[cp_idxNb[cp_idxNb>0].astype(int)-1]
        # print(cp.expand_dims(dEnldXin,3).shape)
        #print("getFi().dEnldXin \n", dEnldXin.shape )

        Fln = cp.sum(cp.expand_dims(dEnldXin,3)*dfeatures,axis=(1,2))

        return Fln

    def loadWij_np(self, f_Wij_np, b_print=False):
        nnWij = np.load(f_Wij_np, allow_pickle=True)
        print('EiNN_cupy.loadWij_np from', f_Wij_np, nnWij.dtype, nnWij.shape)
        return nnWij


#===============================================================================
class NNapiBase:
    """
    NNapiBase, cupy version

    parameters:

    """

    #===========================================================================
    def __init__(self, nn, data_scaler):

        self.maxNb  = nn.maxNb
        self.natoms = nn.natoms
        # self.itp_uc = nn.itp_uc
        self.ds = data_scaler
        self.nn = nn

