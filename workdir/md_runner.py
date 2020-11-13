#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import parameters as pm
# import preparatory_work as ppw
from md_image import MdImage
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen as NVT
from ase.md.nptberendsen import NPTBerendsen as NPT
from ase import units

from calc_lin import calc_lin
from calc_vv import calc_vv

# from minilib.get_util_info import getGpuInfo


class MdRunner():
    
    def __init__( \
                self,
                imageFileDir=pm.mdImageFileDir,\
                isFollow=pm.isFollowMd,\
                calcModel=pm.mdCalcModel,\
                isCheckVar=pm.isMdCheckVar,\
                isReDistribute=pm.isReDistribute,\
                imageIndex=pm.mdStartImageIndex,\
                velocityDistributionModel=pm.velocityDistributionModel,\
                stepTime=pm.mdStepTime,\
                startTemperature=pm.mdStartTemperature,\
                runModel=pm.mdRunModel,\
                endTemperature=pm.mdEndTemperature,\
                nvtTaut=pm.mdNvtTaut,\
                isOnTheFly=pm.isOnTheFlyMd,\
                isTrajAppend=pm.isTrajAppend,\
                isNewMovementAppend=pm.isNewMovementAppend,\
                trajInterval=pm.mdTrajIntervalStepNum,\
                logInterval=pm.mdLogIntervalStepNum,\
                newMovementInterval=pm.mdNewMovementIntervalStepNum,\
                isProfile=pm.isMdProfile
                ):
        
        if calcModel=='lin':
            calc=calc_lin
            # ppw.loadFeatCalcInfo(pm.linModelCalcInfoPath)
            # shutil.copy(pm.linFitInputBakPath,pm.fitInputPath)
        elif calcModel=='vv':
            calc=calc_vv
            # ppw.loadFeatCalcInfo(pm.grrModelCalcInfoPath)
            # shutil.copy(pm.grrFitInputBakPath,pm.fitInputPath)
        else:
            raise NotImplementedError(calcModel+" has't been implemented!")
        
        if isFollow:
            shutil.move(os.path.join(imageFileDir,'last_atom.config'),os.path.join(imageFileDir,'atom.config'))
        
        self.dir=os.path.abspath(imageFileDir)
        self.mdDir=os.path.join(self.dir,'md')
        if not os.path.exists(self.mdDir):
            os.mkdir(self.mdDir)
        elif os.path.isfile(self.mdDir):
            print("Warning: md is a file in the same dir of image config file, this md file will be removed")
            os.remove(self.mdDir)
            os.mkdir(self.mdDir)       
        
        self.atoms=MdImage.fromDir(imageFileDir,calc,isCheckVar,isReDistribute,imageIndex,isProfile)
        if isReDistribute and velocityDistributionModel.lower()=='maxwellboltzmann':
            MaxwellBoltzmannDistribution(self.atoms,startTemperature*units.kB)
        else:
            raise NotImplementedError("Only allow redistribute velocities and apply MaxwellBoltzmannDistribution!")
        
        if runModel.lower()=='nve':
            self.dyn=VelocityVerlet(self.atoms,stepTime*units.fs)
        elif runModel.lower()=='nvt':
            self.dyn=NVT(self.atoms,stepTime*units.fs,endTemperature,nvtTaut*units.fs)
        elif runModel.lower()=='npt':
            self.dyn=NPT(self.atoms,stepTime*units.fs,endTemperature,nvtTaut*units.fs,pressure=1.01325,taup=1.0*1000*units.fs, compressibility=4.57e-5)

        self.isProfile=isProfile
        self.name=os.path.basename(self.dir)    
        self.logFilePath=os.path.join(self.mdDir,self.name+'_log.txt')
        self.trajFilePath=os.path.join(self.mdDir,self.name+'.extxyz')
        self.newMovementPath=os.path.join(self.mdDir,'MOVEMENT')
        self.atomConfigSavePath=os.path.join(self.dir,'last_atom.config')
        self.errorImageLogPath=os.path.join(self.mdDir,self.name+'_errorLog.txt')
        self.profileTxtPath=os.path.join(self.mdDir,self.name+'_profile.txt')
        self.trajInterval=trajInterval
        self.logInterval=logInterval
        self.newMovementInterval=newMovementInterval
        
        if (not isTrajAppend) and os.path.exists(self.trajFilePath):
            os.remove(self.trajFilePath)
        if (not isNewMovementAppend) and os.path.exists(self.newMovementPath):
            os.remove(self.newMovementPath)
        
        self.logFile=open(self.logFilePath,'w')
        self.errorImageLog=open(self.errorImageLogPath,'w')
        if self.isProfile:
            self.profileTxt=open(self.profileTxtPath,'w')
        self.currentStepNum=-1
    
    def runStep(self,nStep=1):
        
        for i in range(nStep):
            
            self.currentStepNum+=1
            self.dyn.run(1)
            
            if self.isProfile:
                profileStr=str(self.currentStepNum+1)+' '+str(self.atoms.calcFeatTime)+' '+str(self.atoms.calcForceTime)
                self.atoms.calcFeatTime=0.0
                self.atoms.calcForceTime=0.0
                if pm.cudaGpuOrder is not None:
                    profileStr=profileStr+' '+str(pm.maxNeighborNum)+' '+str(getGpuInfo()[2][pm.cudaGpuOrder])
                self.profileTxt.write(profileStr+'\n')
            #print(etot,ep,ek)
            
            if self.currentStepNum%self.logInterval==0:
                ek=self.atoms.get_kinetic_energy()
                ep=self.atoms.get_potential_energy()
                etot=ek+ep
                outStr=str(self.currentStepNum+1)+' '+str(etot)+' '+str(ep)+' '+str(ek)
                # if self.atoms.calc==calc_e and self.atoms.isCheckVar:
                #     outStr=outStr+' '+str(self.atoms.calc.flag_of_types.sum())
                #     outStr=outStr+' '+str(self.atoms.calc.var_of_atoms.max())
                #     outStr=outStr+' '+str(self.atoms.calc.var_of_atoms.sum()/len(self.atoms))
                #     if self.atoms.calc.flag_of_types.sum()>0:
                #         errorStr=str(self.currentStepNum+1)+' '+str(self.atoms.calc.flag_of_types.sum())
                #         errorStr=errorStr+' '+str(self.atoms.calc.flag_of_types)
                #         errorStr=errorStr+' '+str(np.where(self.atoms.calc.var_of_atoms>0.2)[0]).replace('\n','')
                #         errorStr=errorStr+' '+str(np.where(self.atoms.neighborNumOfAllAtoms<10)[0]).replace('\n','')
                #         self.errorImageLog.write(errorStr+'\n')                    
                self.logFile.write(outStr+'\n')
            
            if self.currentStepNum%self.trajInterval==0:
                self.atoms.set_positions(self.atoms.get_positions(wrap=True))
                self.atoms.write(self.trajFilePath,append=True)
            
            if self.currentStepNum%self.newMovementInterval==0:
                self.atoms.toAtomConfig(self.newMovementPath,True)

            #if self.currentStepNum%self.newMovementInterval==0:
            #    self.atoms.toTrainMovement(self.newMovementPath,True)

    
    def final(self):
        self.atoms.toAtomConfig(self.atomConfigSavePath)
        self.logFile.close()
        self.errorImageLog.close()
        if self.isProfile:
            self.profileTxt.close()

if __name__=='__main__':   
    input('Press Enter to quit test:')
