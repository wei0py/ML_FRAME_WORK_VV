#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import parameters as pm
import numpy as np
import cupy as cp
import pandas as pd
from image import Image
from ase.cell import Cell
from ase.atoms import Atoms
from ase.constraints import (voigt_6_to_full_3x3_stress,
                             full_3x3_to_voigt_6_stress)
import math                        

# from calc_feature_nn import calc_feature
from calc_ftype1 import calc_ftype1
from calc_ftype2 import calc_ftype2
from nn_model_cupy import NNapiBase #,EiNN_cupy
from calc_rep import calc_rep
# from minilib.get_util_info import getGpuInfo

class MdImage(Atoms,Image,NNapiBase):

    def __init__(self):
        pass


    @staticmethod
    def fromImage(anImage,nn,data_scaler,isCheckVar=False,isReDistribute=True):
        
        self=MdImage()
        
        Image.__init__(self,
        atomTypeList=anImage.atomTypeList,
        cell=anImage.cupyCell,
        pos=anImage.pos,
        isOrthogonalCell=anImage.isOrthogonalCell,
        isCheckSupercell=anImage.isCheckSupercell,
        atomTypeSet=anImage.atomTypeSet,
        atomCountAsType=anImage.atomCountAsType,
        atomCategoryDict=anImage.atomCategoryDict,
        force=anImage.force,
        energy=anImage.energy,
        velocity=anImage.velocity,
        ep=anImage.ep,
        dE=anImage.dE)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(self.pos),
        cell=Cell(cp.asnumpy(self.cupyCell)),
        pbc=True,
        numbers=self.atomTypeList)

        NNapiBase.__init__(self,nn=nn,data_scaler=data_scaler)
        
        self.isCheckVar=isCheckVar
        if pm.add_force:
            self.add_force=np.loadtxt('add_force')
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)

        self.isNewStep=True
        return self


    @staticmethod
    def fromAtoms(anAtoms):
        pass

    @staticmethod
    def fromAtomConfig(atomConfigPath,nn,data_scaler,isCheckVar=False,isReDistribute=True):
        
        numOfAtoms=int(open(atomConfigPath,'r').readline())
        cell=cp.array(pd.read_csv(atomConfigPath,delim_whitespace=True,header=None,skiprows=2,nrows=3))
        data=pd.read_csv(atomConfigPath,delim_whitespace=True,header=None,skiprows=6,nrows=numOfAtoms)
        atomTypeList=list(data[0])
        pos=cp.array(data.iloc[:,1:4])
        
        self=MdImage()
        
        Image.__init__(self,
        atomTypeList=atomTypeList,
        cell=cell,
        pos=pos)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(pos),
        cell=Cell(cp.asnumpy(cell)),
        pbc=True,
        numbers=self.atomTypeList)

        NNapiBase.__init__(self,nn=nn,data_scaler=data_scaler)
        
        self.isCheckVar=isCheckVar
        if pm.add_force:
            self.add_force=np.loadtxt('add_force')
        # calc_feature.set_paths(pm.fitModelDir)
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)
        self.isNewStep=True
        return self


    @staticmethod
    def fromMovement(movementPath,nn,data_scaler,isCheckVar=False,isReDistribute=True,imageIndex=0):
        
        with open(movementPath,'r') as sourceFile:
            numOfAtoms=int(sourceFile.readline().split()[0])
        
        with open(movementPath,'r') as sourceFile:
            currentIndex=-1
            while True:
                line=sourceFile.readline()
                if not line:
                    raise EOFError("The Movement file end, there is only "+str(currentIndex+1)+\
                                   " images, and the "+str(imageIndex+1)+"th image has been choosen!")
                if "Iteration" in line:
                    currentIndex+=1
                
                if currentIndex<imageIndex:
                    continue
                else:
                    cell=cp.zeros((3,3))
                    atomTypeList=[]
                    pos=cp.zeros((numOfAtoms,3))
                    while True:
                        line=sourceFile.readline()
                        if "Lattice" in line:
                            break
                    for i in range(3):
                        L=sourceFile.readline().split()
                        for j in range(3):
                            cell[i,j]=float(L[j])
                    line=sourceFile.readline()
                    for i in range(numOfAtoms):
                        L=sourceFile.readline().split()
                        atomTypeList.append(int(L[0]))
                        for j in range(3):
                            pos[i,j]=float(L[j+1])
                    break
        
        self=MdImage()
        Image.__init__(self,
        atomTypeList=atomTypeList,
        cell=cell,
        pos=pos)
        
        
        Atoms.__init__(self,
        scaled_positions=cp.asnumpy(pos),
        cell=Cell(cp.asnumpy(cell)),
        pbc=True,
        numbers=self.atomTypeList)

        NNapiBase.__init__(self,nn=nn,data_scaler=data_scaler)
        
        self.isCheckVar=isCheckVar
        if pm.add_force:
            self.add_force=np.loadtxt('add_force')
        
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                calc_ftype1.load_model()
                calc_ftype1.set_image_info(np.array(atomTypeList),True)
            if pm.use_Ftype[i]==2:
                calc_ftype2.load_model()
                calc_ftype2.set_image_info(np.array(atomTypeList),True)

        self.isNewStep=True
        return self
            


    @staticmethod
    def fromDir(dirPath,nn,data_scaler,isCheckVar=False,isReDistribute=True,imageIndex=0,isProfile=None):
        
        if isProfile==None:
            isProfile=pm.isMdProfile
        
        dirList=os.listdir(dirPath)
        if 'atom.config' in dirList and os.path.isfile(os.path.join(dirPath,'atom.config')):
            atomConfigFilePath=os.path.join(dirPath,'atom.config')
            self=MdImage.fromAtomConfig(atomConfigFilePath,nn,data_scaler,isCheckVar,isReDistribute)
            self.dir=os.path.abspath(dirPath)            
            self.isProfile=isProfile
            if self.isProfile:
                self.calcFeatTime=0.0
                self.calcForceTime=0.0
            return self
        elif 'MOVEMENT' in dirList and os.path.isfile(os.path.join(dirPath,'MOVEMENT')):
            movementPath=os.path.join(dirPath,'MOVEMENT')
            self=MdImage.fromMovement(movementPath,nn,data_scaler,isCheckVar,isReDistribute,imageIndex)
            self.dir=os.path.abspath(dirPath)
            self.isProfile=isProfile
            if self.isProfile:
                self.calcFeatTime=0.0
                self.calcForceTime=0.0
            return self
        else:
            raise ValueError("There is no atom.config or MOVEMENT in this dir")



    def toAtomConfig(self,atomConfigFilePath,isAppend=False,isApplyConstraint=False):
        if isAppend:
            openMode='a'
        else:
            openMode='w'
        energies=self.get_potential_energies()
        forces=self.get_forces()
        with open(atomConfigFilePath,openMode) as atomConfigFile:
            atomConfigFile.write(str(len(self))+'\n')
            atomConfigFile.write('LATTICE\n')
            for i in range(3):
                atomConfigFile.write(str(float(self.cell[i,0]))+'  '+str(float(self.cell[i,1]))+'  '+str(float(self.cell[i,2]))+'  \n')
            atomConfigFile.write('POSITION\n')
            if not isApplyConstraint:
                lineTail='  1  1  1  '
                sPos=self.get_scaled_positions(True)
                for i in range(len(self)):
                    atomConfigFile.write(str(self.atomTypeList[i])+'  '+str(float(sPos[i,0]))+'  '+str(float(sPos[i,1]))+'  '+\
                    str(float(sPos[i,2]))+'  '+lineTail+str(energies[i])+'  '+str(forces[i,0])+'  '+str(forces[i,1])+'  '+str(forces[i,2])+'\n')
                    
            atomConfigFile.write('-'*80+'\n')

    def toTrainMovement(self,atomConfigFilePath,isAppend=False,isApplyConstraint=False):
        if isAppend:
            openMode='a'
        else:
            openMode='w'
        energies=self.get_potential_energies()
        forces=self.get_forces()
        velocities=self.get_velocities()
        ek=np.sum(self.get_kinetic_energy())
        ep=np.sum(energies)
        etot=ek+ep
        with open(atomConfigFilePath,openMode) as atomConfigFile:
            atomConfigFile.write(str(len(self))+'  atoms,Iteration (fs) =    0.2000000000E+01, Etot,Ep,Ek (eV) =   '+str(etot)+'  '+str(ep)+'   '+str(ek)+'\n')
            atomConfigFile.write(' Lattice vector (Angstrom)\n')
            for i in range(3):
                atomConfigFile.write(str(float(self.cell[i,0]))+'  '+str(float(self.cell[i,1]))+'  '+str(float(self.cell[i,2]))+'  \n')
            atomConfigFile.write(' Position (normalized), move_x, move_y, move_z\n')
            if not isApplyConstraint:
                lineTail='  1  1  1  '
                sPos=self.get_scaled_positions(True)
                for i in range(len(self)):
                    atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(float(sPos[i,0]))+'    '+str(float(sPos[i,1]))+'    '+\
                    str(float(sPos[i,2]))+'   '+lineTail+'\n')        
            atomConfigFile.write('Force (-force, eV/Angstrom)\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(-forces[i,0])+'    '+str(-forces[i,1])+'    '+str(-forces[i,2])+'\n')
            atomConfigFile.write(' Velocity (bohr/fs)\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(velocities[i,0])+'    '+str(velocities[i,1])+'    '+str(velocities[i,2])+'\n')
            atomConfigFile.write('Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  0.0\n')
            for i in range(len(self)):
                atomConfigFile.write(str(self.atomTypeList[i])+'    '+str(energies[i])+' \n')
            atomConfigFile.write('-'*80+'\n')                
            
    # def calAllNeighborStruct(self,isSave=False,isCheckFile=True,rMin=pm.rMin,rCut=pm.Rc_M):
    #     Image.calAllNeighborStruct(self,isSave,isCheckFile,rMin,rCut)
    #     if not pm.isFixedMaxNeighborNumForMd:
    #         self.preMaxNeighborNum=0
    #     pm.maxNeighborNum=int(self.neighborNumOfAllAtoms.max())
    #     if pm.maxNeighborNum!=self.preMaxNeighborNum:
    #         mempool = cp.get_default_memory_pool()
    #         mempool.free_all_blocks()
    #     self.preMaxNeighborNum=pm.maxNeighborNum
    
    def set_pos_cell(self):
        
        scaled_positions=cp.array(self.get_scaled_positions(True))
        cell=cp.array(np.array(self.cell))
        if ((self.cupyCell==cell).all() and (self.pos==scaled_positions).all()):
            return
        else:
            self.cupyCell=cell
            self.pos=scaled_positions
            self.isNewStep=True
        
    
    def set_positions(self, newpositions, apply_constraint=True):
        Atoms.set_positions(self,newpositions,apply_constraint)
        self.set_pos_cell()

    def calc_feat(self):
        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)
        for i in range(len(pm.use_Ftype)):
            if pm.use_Ftype[i]==1:
                # start=time.time()
                calc_ftype1.gen_feature(cell,pos)
                # print("only feat2b time: ",time.time()-start)
                feat_tmp=np.array(calc_ftype1.feat).transpose()
                dfeat_tmp=np.array(calc_ftype1.dfeat).transpose(1,2,0,3)
                list_neigh=calc_ftype1.list_neigh_alltypem
                # nblist = np.array(calc_ftype1.list_neigh_alltypem).transpose().astype(int)
                num_neigh = calc_ftype1.num_neigh_alltypem
            if pm.use_Ftype[i]==2:
                # start=time.time()
                calc_ftype2.gen_feature(cell,pos)
                # print("only feat3b time: ",time.time()-start)
                feat_tmp=np.array(calc_ftype2.feat).transpose()
                dfeat_tmp=np.array(calc_ftype2.dfeat).transpose(1,2,0,3)
                list_neigh=calc_ftype2.list_neigh_alltypem
                # nblist = np.array(calc_ftype2.list_neigh_alltypem).transpose().astype(int)
                num_neigh = calc_ftype2.num_neigh_alltypem
            if i==0:
                feat=feat_tmp
                dfeat=dfeat_tmp
            else:
                feat=np.concatenate((feat,feat_tmp),axis=1)
                dfeat=np.concatenate((dfeat,dfeat_tmp),axis=2)

        # num_neigh_alltype=np.array(calc_feature.num_neigh_alltype)
        
       
        return feat,dfeat,list_neigh,num_neigh #nblist

    def calcEnergiesForces(self):
        
        start=time.time()
        
        '''
        if pm.cudaGpuOrder is not None:
            print("Before calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)
        '''
        if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
            del self.neighborDistanceVectArrayOfAllAtoms
#TODO:
        
        feat,dfeat,list_neigh,num_neigh=self.calc_feat()
        # print(cell)
        print("cal feat time: ",time.time()-start)
        # print(pos)
        # print(np.shape(dfeat))
        # start2=time.time()
        nblist = np.array(list_neigh).transpose().astype(int)
        itype=self.numbers
        feat_scaled = self.ds.pre_feat(feat, itype)
                
        engy_out = self.nn.getEi(feat_scaled, itype)
        Ep = self.ds.post_engy(engy_out, itype)
        

        dfeat_scaled = self.ds.pre_dfeat(dfeat, itype[:,np.newaxis], nblist)
        f_out    = self.nn.getFi(feat_scaled, dfeat_scaled, nblist, itype) #TODO:
        Fp = self.ds.post_fors(f_out, itype)

        # print("GPU energy force time: ",time.time()-start2)
        # TODO:
        # start3=time.time()
        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)
        wp_atom=np.asfortranarray(np.array(pm.fortranFitAtomRepulsingEnergies).transpose())
        rad_atom=np.asfortranarray(np.array(pm.fortranFitAtomRadii).transpose())
        iatom_type=np.zeros_like(itype)
        for m in range(len(itype)):
            iatom_type[m]=pm.atomType.index(itype[m])
        iatom_type=np.asfortranarray(iatom_type.transpose())


        calc_rep.calc_replusive(num_neigh,list_neigh,cell,pos,iatom_type,rad_atom,wp_atom)
        dE=cp.array(calc_rep.energy).reshape((-1,1))
        dF=cp.array(calc_rep.force).transpose()
        Ep=Ep+dE
        Fp=Fp+dF
        calc_rep.deallo()

        # for i in range(len(itype)):
        #     rad1=rad_atom[iatom_type[i]]
        #     dE=0.0
        #     dFx=0.0
        #     dFy=0.0
        #     dFz=0.0
        #     for j in nblist[i,:]:
        #         if (j-1) != i and j > 0:
        #             rad2 = rad_atom[iatom_type[j-1]]
        #             rad=rad1+rad2
        #             dx1=(pos[j-1,0]-pos[i,0])%1.0
        #             if (abs(dx1-1) < abs(dx1)):
        #                 dx1=dx1-1
        #             dx2=(pos[j-1,1]-pos[i,1])%1.0
        #             if (abs(dx2-1) < abs(dx2)):
        #                 dx2=dx2-1
        #             dx3=(pos[j-1,2]-pos[i,2])%1.0
        #             if (abs(dx3-1) < abs(dx3)):
        #                 dx3=dx3-1
        #             dx=cell[0,0]*dx1+cell[1,0]*dx2+cell[2,0]*dx3
        #             dy=cell[0,1]*dx1+cell[1,1]*dx2+cell[2,1]*dx3
        #             dz=cell[0,2]*dx1+cell[1,2]*dx2+cell[2,2]*dx3
        #             # dd=math.sqrt(dx**2+dy**2+dz**2)
        #             # [dx,dy,dz]=self.get_distance(i,j-1,mic=True,vector=True)
        #             dd=math.sqrt(dx**2+dy**2+dz**2)
        #             if(dd < 2*rad):
        #                 w22=math.sqrt(wp_atom[iatom_type[i]]*wp_atom[iatom_type[j-1]])
        #                 yy=math.pi*dd/(4*rad)
        #                 dE=dE+0.5*4*w22*(rad/dd)**12*math.cos(yy)**2
        #                 dEdd=4*w22*(-12*(rad/dd)**12/dd*math.cos(yy)**2-(math.pi/(2*rad))*math.cos(yy)*math.sin(yy)*(rad/dd)**12)

        #                 dFx=dFx-dEdd*dx/dd       #! note, -sign, because dx=d(j)-x(i)
        #                 dFy=dFy-dEdd*dy/dd
        #                 dFz=dFz-dEdd*dz/dd
        #     Ep[i]=Ep[i]+dE
        #     Fp[i,0]=Fp[i,0]+dFx
        #     Fp[i,1]=Fp[i,1]+dFy
        #     Fp[i,2]=Fp[i,2]+dFz
            # if abs(dFx) > 0.5 or abs(dFy)>0.5 or abs(dFz)>0.5:
            #     print(i+1,'  ',dFx,'  ',dFy,'  ',dFz)
            # print('dFx:',dFx)
            # print('dFy:',dFy)
            # print('dFz:',dFz)
        # print("repulsive time: ",time.time()-start3)
        if pm.add_force:
            add_force=self.add_force
            if int(add_force[0,1])==0:
                for i in range(1,len(add_force)):                
                    Fp[add_force[i,0]-1,0]=Fp[add_force[i,0]-1,0]+(add_force[i,1]-1)*add_force[i,2]
            if int(add_force[0,1])==2:
                for i in range(1,len(add_force)):                
                    Fp[add_force[i,0]-1,2]=Fp[add_force[i,0]-1,2]+(add_force[i,1]-1)*add_force[i,2]
            if int(add_force[0,1])==1:
                for i in range(1,len(add_force)):                
                    Fp[add_force[i,0]-1,1]=Fp[add_force[i,0]-1,1]+(add_force[i,1]-1)*add_force[i,2]
        
        self.energies=cp.asnumpy(Ep).reshape(-1)
        self.etot=cp.sum(Ep)
        self.forces = - cp.asnumpy(Fp)

        #print("repulsive and add force time: ",time.time()-start3)

        #print("cal feat time: ",time.time()-start)
        if self.isProfile:
            self.calcFeatTime+=time.time()-start
            star=time.time()

        # istatCalc=int(self.calc.istat)
        # errorMsgCalc=calc_lin.error_msg.tostring().decode('utf-8').rstrip()
        # if istatCalc!=0:
        #     raise ValueError(errorMsgCalc)
        self.isNewStep=False
        
        
        '''
        if pm.cudaGpuOrder is not None:
            print("After calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)   
        '''
        #print("cal force time: ",time.time()-start)
        if self.isProfile:
            self.calcForceTime+=time.time()-start
            print(self.calcFeatTime,self.calcForceTime)
        
    def get_potential_energy(self):
        
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.etot
    
    def get_potential_energies(self):
        
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.energies
        
    def get_forces(self,apply_constraint=True, md=False):
        '''
        暂时还未考虑constraint!
        '''
        self.set_pos_cell()
        if self.isNewStep:
            # print('calc in ',sys._getframe().f_code.co_name)
            self.calcEnergiesForces()        
        return self.forces


    def calc_only_energy(self):
        
        start=time.time()
        self.set_pos_cell()
        '''
        if pm.cudaGpuOrder is not None:
            print("Before calc feat, used gpu memory and maxNeighborNum:")
            print(getGpuInfo()[2][pm.cudaGpuOrder],pm.maxNeighborNum)
        '''
        if hasattr(self,'neighborDistanceVectArrayOfAllAtoms'):
            del self.neighborDistanceVectArrayOfAllAtoms
        # cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        # pos=np.asfortranarray(self.get_scaled_positions(True).T)
        # feat,_,nblist=self.calc_feat()
        feat,_,list_neigh,num_neigh=self.calc_feat()
        # print(cell)
        # print(pos)
        # print(np.shape(feat))
        itype=self.numbers
        feat_scaled = self.ds.pre_feat(feat, itype)
                
        engy_out = self.nn.getEi(feat_scaled, itype)
        Ep = self.ds.post_engy(engy_out, itype)

        cell=np.asfortranarray(cp.asnumpy(self.cupyCell.T))
        pos=np.asfortranarray(self.get_scaled_positions(True).T)
        wp_atom=np.asfortranarray(np.array(pm.fortranFitAtomRepulsingEnergies).transpose())
        rad_atom=np.asfortranarray(np.array(pm.fortranFitAtomRadii).transpose())
        iatom_type=np.zeros_like(itype)
        for m in range(len(itype)):
            iatom_type[m]=pm.atomType.index(itype[m])
        iatom_type=np.asfortranarray(iatom_type.transpose())


        calc_rep.calc_rep(num_neigh,list_neigh,cell,pos,iatom_type,rad_atom,wp_atom)
        dE=np.array(calc_rep.energy).reshape((-1,1))
        # dF=np.array(calc_rep.force).transpose()
        Ep=Ep+dE
        # Fp=Fp+dF
        calc_rep.deallo()
        # if pm.add_force:
        #     add_force=np.loadtxt('add_force')
        #     for i in range(1,len(add_force)):                
        #         Fp[add_force[i,0]-1,0]=Fp[add_force[i,0]-1,0]+(add_force[i,1]-1)*add_force[i,2]
                
        #print("cal feat time: ",time.time()-start)
        if self.isProfile:
            self.calcFeatTime+=time.time()-start
            star=time.time()

        # self.isNewStep=False
        
        if self.isProfile:
            self.calcForceTime+=time.time()-start
            print(self.calcFeatTime,self.calcForceTime)
        
        return cp.sum(Ep)


    def calculate_numerical_stress(self, atoms,d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""

        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            print(self.cell)
            print(self.cupyCell)
        
            eplus = atoms.calc_only_energy()
            print(self.cell)
            print(self.cupyCell)

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            print(self.cell)
            print(self.cupyCell)
            # self.isNewStep=True
            eminus = atoms.calc_only_energy()
            print(self.cell)
            print(self.cupyCell)

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            print(self.cell)
            print(self.cupyCell)
            # self.isNewStep=True
            eplus = atoms.calc_only_energy()
            print(self.cell)
            print(self.cupyCell)

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            print(self.cell)
            print(self.cupyCell)
            # self.isNewStep=True
            eminus = atoms.calc_only_energy()
            print(self.cell)
            print(self.cupyCell)

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)
        self.set_pos_cell()
        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress


    def get_stress(self, voigt=True, apply_constraint=True,
                   include_ideal_gas=False):
        """Calculate stress tensor.

        Returns an array of the six independent components of the
        symmetric stress tensor, in the traditional Voigt order
        (xx, yy, zz, yz, xz, xy) or as a 3x3 matrix.  Default is Voigt
        order.

        The ideal gas contribution to the stresses is added if the
        atoms have momenta and ``include_ideal_gas`` is set to True.
        """

        # if self._calc is None:
        #     raise RuntimeError('Atoms object has no calculator.')

        stress = self.calculate_numerical_stress(self,voigt=voigt)
        shape = stress.shape

        if shape == (3, 3):
            # Convert to the Voigt form before possibly applying
            # constraints and adding the dynamic part of the stress
            # (the "ideal gas contribution").
            stress = full_3x3_to_voigt_6_stress(stress)
        else:
            assert shape == (6,)

        if apply_constraint:
            for constraint in self.constraints:
                if hasattr(constraint, 'adjust_stress'):
                    constraint.adjust_stress(self, stress)

        # Add ideal gas contribution, if applicable
        if include_ideal_gas and self.has('momenta'):
            stresscomp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
            p = self.get_momenta()
            masses = self.get_masses()
            invmass = 1.0 / masses
            invvol = 1.0 / self.get_volume()
            for alpha in range(3):
                for beta in range(alpha, 3):
                    stress[stresscomp[alpha, beta]] -= (
                        p[:, alpha] * p[:, beta] * invmass).sum() * invvol

        if voigt:
            return stress
        else:
            return voigt_6_to_full_3x3_stress(stress)
            # return stress

    # def get_stresses(self, include_ideal_gas=False, voigt=True):
    #     """Calculate the stress-tensor of all the atoms.

    #     Only available with calculators supporting per-atom energies and
    #     stresses (e.g. classical potentials).  Even for such calculators
    #     there is a certain arbitrariness in defining per-atom stresses.

    #     The ideal gas contribution to the stresses is added if the
    #     atoms have momenta and ``include_ideal_gas`` is set to True.
    #     """
    #     # if self._calc is None:
    #     #     raise RuntimeError('Atoms object has no calculator.')
    #     stresses = self._calc.get_stresses(self)

    #     # make sure `stresses` are in voigt form
    #     if np.shape(stresses)[1:] == (3, 3):
    #         stresses_voigt = [full_3x3_to_voigt_6_stress(s) for s in stresses]
    #         stresses = np.array(stresses_voigt)

    #     # REMARK: The ideal gas contribution is intensive, i.e., the volume
    #     # is divided out. We currently don't check if `stresses` are intensive
    #     # as well, i.e., if `a.get_stresses.sum(axis=0) == a.get_stress()`.
    #     # It might be good to check this here, but adds computational overhead.

    #     if include_ideal_gas and self.has('momenta'):
    #         stresscomp = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
    #         if hasattr(self._calc, 'get_atomic_volumes'):
    #             invvol = 1.0 / self._calc.get_atomic_volumes()
    #         else:
    #             invvol = self.get_global_number_of_atoms() / self.get_volume()
    #         p = self.get_momenta()
    #         invmass = 1.0 / self.get_masses()
    #         for alpha in range(3):
    #             for beta in range(alpha, 3):
    #                 stresses[:, stresscomp[alpha, beta]] -= (
    #                     p[:, alpha] * p[:, beta] * invmass * invvol)
    #     if voigt:
    #         return stresses
    #     else:
    #         stresses_3x3 = [voigt_6_to_full_3x3_stress(s) for s in stresses]
    #         return np.array(stresses_3x3)
        
     

if __name__=='__main__':   
    input('Press Enter to quit test:')
