#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Fri Aug 24 00:11:27 PDT 2018
@author: lingmiao@lbl.gov at Prof. LinWang Wang's group
"""

import parameters as pm
# if not pm.istrain:
#     import cupy as cp
# else:
import numpy as cp

import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

#================================================================================

def mae(Enn, Edft):
    return cp.mean(cp.abs(Enn-Edft))

def mse(Enn, Edft):
    return cp.sqrt(cp.mean((Enn-Edft)**2))

def mse4(Enn, Edft, scl=10):
    return cp.mean((scl*(Enn-Edft))**4)

#================================================================================

class MinMaxScaler:
    ''' a*x +b = x_scaled like sklearn's MinMaxScaler
        note cp.atleast_2d and self.a[xmax==xmin] = 0
    '''
    def __init__(self, feature_range=(0,1)):
        self.fr = feature_range
        self.a = 0
        self.b = 0

    def fit_transform(self, x):

        if len(x) == 0:
            self.a = 0
            self.b = 0
            return x
        
        x = cp.atleast_2d(x)

        xmax = x.max(axis=0)
        xmin = x.min(axis=0)
#TODO:
        self.a = (self.fr[1] -self.fr[0]) / (xmax-xmin)
        self.a[xmax==xmin] = 0  # important !!!
        self.b = self.fr[0] - self.a*xmin
        # self.a = cp.ones(xmax.shape)
        # self.b = cp.zeros(xmin.shape)

        return self.transform(x)

    def transform(self, x):
        x = cp.atleast_2d(x)
        return self.a*x +self.b

    def inverse_transform(self, y):
        y = cp.atleast_2d(y)
        return (y -self.b) /self.a

#================================================================================

class DataScaler:
    
    def __init__(self):
        self.feat_scaler=MinMaxScaler(feature_range=(0,1))
        self.feat_a = None
        self.engy_scaler=MinMaxScaler(feature_range=(0,1))
        self.engy_a = None

        return

    #===========================================================================
    
    def get_scaler(self, f_feat, f_ds, b_save=True):
        
        from prepare import r_feat_csv

        itypes,feat,engy = r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        
        _ = self.feat_scaler.fit_transform(feat)
        _ = self.engy_scaler.fit_transform(engy)
        
        feat_b      = self.feat_scaler.transform(cp.zeros((1, feat.shape[1])))    
        self.feat_a = self.feat_scaler.transform(cp.ones((1, feat.shape[1]))) - feat_b
        engy_b      = self.engy_scaler.transform(0)
        self.engy_a = self.engy_scaler.transform(1) - engy_b

        #return self.feat_scaler

    #===========================================================================
    
    def pre_feat(self, feat):
        return self.feat_scaler.transform(feat)

    # def pre_dfeat(self, dfeat):
    #     return self.feat_a[cp.newaxis,cp.newaxis,:,:] * dfeat
        # return self.feat_a[:,:,cp.newaxis] * dfeat

#================================================================================

class DataScalers:
    '''
    The wrapper for multiple elements. Generally a dictionary with data scalers for each element.
    Notice the 's'. It is important.
    '''

    def __init__(self, f_ds, f_feat, load=False):

        self.scalers = {}
        self.feat_as = {}
        self.engy_as = {}
        
        for i in range(pm.ntypes):
            self.scalers[pm.atomType[i]] = DataScaler()

        from os import path

        if load and path.isfile(f_ds):
            self.loadDSs_np(f_ds)
        elif path.isfile(f_feat):
            self.get_scalers(f_feat, f_ds, b_save=True)
        else:
            exit(["===Error in DataScaler, don't find ", f_ds, f_feat, '==='])

        return

    #===========================================================================

    def get_scalers(self, f_feat, f_ds, b_save=True):
        
        from prepare import r_feat_csv

        itypes,feat,engy = r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        print('=DS.get_scaler ', f_feat, 'engy.shape, feat.dtype', engy.shape, engy.dtype)
        print('=DS.get_scaler ', f_feat, 'itypes.shape, feat.dtype', itypes.shape, itypes.dtype)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            subfeat = feat[itypes == itype]
            subengy = engy[itypes == itype]
            _ = self.scalers[itype].feat_scaler.fit_transform(subfeat)
            _ = self.scalers[itype].engy_scaler.fit_transform(subengy)
            feat_b = self.scalers[itype].feat_scaler.transform(cp.zeros((1, subfeat.shape[1])))
            engy_b = self.scalers[itype].engy_scaler.transform(cp.zeros((1, subengy.shape[1])))
            self.feat_as[itype] = self.scalers[itype].\
                                 feat_scaler.transform(cp.ones((1, subfeat.shape[1]))) - feat_b
            self.engy_as[itype] = self.scalers[itype].\
                                 engy_scaler.transform(cp.ones((1, subengy.shape[1]))) - engy_b

        if b_save:
            self.save2np(f_ds)
    
        #return self.feat_scalers

    #===========================================================================

    def pre_engy(self, engy, itypes):
        # engy_scaled = cp.zeros_like(engy)
        # for i in range(pm.ntypes):
        #     itype = pm.atomType[i]
        #     engy_scaled[itypes == itype] = self.scalers[itype].\
        #         engy_scaler.transform(engy[itypes == itype])
        return engy

    def post_engy(self, engy_scaled, itypes):
        # engy_orig = cp.zeros_like(engy_scaled)
        # for i in range(pm.ntypes):
        #     itype = pm.atomType[i]
        #     engy_orig[itypes == itype] = self.scalers[itype].\
        #         engy_scaler.inverse_transform(engy_scaled[itypes == itype])
        return engy_scaled

    def pre_fors(self, fors, itypes):
        # fors_scaled = cp.zeros_like(fors)
        # for i in range(pm.ntypes):
        #     itype = pm.atomType[i]
        #     fors_scaled[itypes == itype] = fors[itypes == itype] * self.engy_as[itype][0]
        return fors

    def post_fors(self, fors_scaled, itypes):
        # fors_orig = cp.zeros_like(fors_scaled)
        # for i in range(pm.ntypes):
        #     itype = pm.atomType[i]
        #     fors_orig[itypes == itype] = fors_scaled[itypes == itype] / self.engy_as[itype][0]
        return fors_scaled

    def pre_feat(self, feat, itypes):
        feat_scaled = cp.zeros(feat.shape)
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaled[itypes == itype] = self.scalers[itype].\
                                           feat_scaler.transform(feat[itypes == itype])
        return feat_scaled

    def pre_dfeat(self, dfeat, itypes, nblt):
        dfeat_scaled = cp.zeros(dfeat.shape)
        #print(dfeat_scaled.shape)
        #print(dfeat.shape)
        # print(itypes.shape)
        # print(nblt.shape)
        #print((itypes[nblt-1]==26).shape)
        #print(self.feat_as[26].shape)
        natoms=dfeat.shape[0]
        max_nb=dfeat.shape[1]
        featnum=dfeat.shape[2]
       
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            #print(itype)
            #print(dfeat_scaled[itypes.squeeze()[nblt-1] == itype].shape)
            #print(self.feat_as[itype][:,:,cp.newaxis].shape)
            # for l in range(natoms):
            #     for k in range(max_nb):
            #         if itypes[nblt[l,k]-1] == itype and nblt[l,k]>0:
            #             # print(l,k)
            #             # dfeat_scaled[l,k,:,:]=dfeat[l,k,:,:]*self.feat_as[itype][0,:,cp.newaxis]
            #             for m in range(featnum):
            #                 dfeat_scaled[l,k,m,0]=dfeat[l,k,m,0]*self.feat_as[itype][0,m]
            #                 dfeat_scaled[l,k,m,1]=dfeat[l,k,m,1]*self.feat_as[itype][0,m]
            #                 dfeat_scaled[l,k,m,2]=dfeat[l,k,m,2]*self.feat_as[itype][0,m]


            dfeat_scaled[(itypes.squeeze()[nblt-1] == itype) & (nblt>0)] = self.feat_as[itype][:,:,cp.newaxis]\
                                            *dfeat[(itypes.squeeze()[nblt-1]==itype)&(nblt>0)]
        return dfeat_scaled

    def save2np(self, f_npfile):
        dsnp = []
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            feat_scaler = self.scalers[itype].feat_scaler
            engy_scaler = self.scalers[itype].engy_scaler
            dsnp.append(np.array(feat_scaler.fr))
            dsnp.append(np.array(feat_scaler.a))
            dsnp.append(np.array(feat_scaler.b))
            dsnp.append(np.array(self.feat_as[itype]))
            dsnp.append(np.array(engy_scaler.fr))
            dsnp.append(np.array(engy_scaler.a))
            dsnp.append(np.array(engy_scaler.b))
            dsnp.append(np.array(self.engy_as[itype]))
        dsnp = np.array(dsnp)
        np.save(f_npfile, dsnp)
        print('DataScaler.save2np to', f_npfile, dsnp.dtype, dsnp.shape)
        return

    def loadDSs_np(self, f_npfile):
        dsnp = np.load(f_npfile, allow_pickle=True)

        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            self.scalers[itype].feat_scaler.fr = cp.asarray(dsnp[8*i+0])
            self.scalers[itype].feat_scaler.a  = cp.asarray(dsnp[8*i+1])
            self.scalers[itype].feat_scaler.b  = cp.asarray(dsnp[8*i+2])
            self.feat_as[itype]         = cp.asarray(dsnp[8*i+3])
            self.scalers[itype].engy_scaler.fr = cp.asarray(dsnp[8*i+4])
            self.scalers[itype].engy_scaler.a  = cp.asarray(dsnp[8*i+5])
            self.scalers[itype].engy_scaler.b  = cp.asarray(dsnp[8*i+6])
            self.engy_as[itype]         = cp.asarray(dsnp[8*i+7])


        print('DataScaler.loadDS_np from', f_npfile, dsnp.dtype, dsnp.shape)
        #for i in range(dsnp.shape[0]):
        #    print("dsnp[i]",i, dsnp[i].shape)
        return
