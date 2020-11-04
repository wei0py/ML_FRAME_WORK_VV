import os
import parameters as pm
import numpy as np
import cupy as cp


def collectAllSourceFiles(workDir=pm.trainSetDir,sourceFileName='MOVEMENT'):
    '''
    搜索工作文件夹，得到所有MOVEMENT文件的路径，并将之存储在pm.sourceFileList中
    
    Determine parameters:
    ---------------------
    pm.sourceFileList:            List对象，罗列了所有MOVEMENT文件的路径        
    '''
    if not os.path.exists(workDir):
        raise FileNotFoundError(workDir+'  is not exist!')
    for path,dirList,fileList in os.walk(workDir):
        if sourceFileName in fileList:
            pm.sourceFileList.append(os.path.abspath(path))


def savePath(featSaveForm='C'):
    '''
    save path to file
    '''
    featSaveForm=featSaveForm.upper()
    pm.numOfSystem=len(pm.sourceFileList)
    with open(pm.fbinListPath,'w') as fbinList:
        fbinList.write(str(pm.numOfSystem)+'\n')
        fbinList.write(str(os.path.abspath(pm.trainSetDir))+'\n')
        for system in pm.sourceFileList:
            fbinList.write(str(system)+'\n')

def combineMovement():
    '''
    combine MOVEMENT file
    '''
    with open(os.path.join(os.path.abspath(pm.trainSetDir),'MOVEMENTall'), 'w') as outfile:     
        # Iterate through list 
        for names in pm.sourceFileList:     
            # Open each file in read mode 
            with open(os.path.join(os.path.abspath(names),'MOVEMENT')) as infile:     
                # read the data from file1 and 
                # file2 and write it in file3 
                outfile.write(infile.read()) 
    
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

def movementUsed():
    '''
    index images not used
    '''
    badImageNum=0
    for names in pm.sourceFileList:
        image=np.loadtxt(os.path.join(os.path.abspath(names),'info.txt'))
        badimage=image[3:]
        badImageNum=badImageNum+len(badimage)
    
    with open(os.path.join(os.path.abspath(pm.trainSetDir),'imagesNotUsed'), 'w') as outfile:     
        # Iterate through list 
        outfile.write(str(badImageNum)+'  \n')
        index=0
        
        for names in pm.sourceFileList:
            image=np.loadtxt(os.path.join(os.path.abspath(names),'info.txt'))
            badimage=image[3:]
            numOfImage=image[2]
            for i in range(len(badimage)):
                outfile.write(str(badimage[i]+index)+'\n')
            index=index+numOfImage
    


def writeGenFeatInput():
    '''
        #    1       ! ntype
        #    16      ! iat-type1
        #    5.5,11.0,5.0,2,0.2,0.5,0.5    ! Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2
        #    24,3,3  ! n2b, n3b1, n3b2
        #    100    ! m_neigh
        #    0.3    ! E_tolerance    
        #    3      ! iflag_ftype
    '''
    with open(pm.GenFeatInputPath,'w') as GenFeatInput:
        GenFeatInput.write(str(len(pm.atomType))+'          ! ntype \n')
        for i in range(pm.atomTypeNum):
            GenFeatInput.write(str(pm.atomType[i])+'          ! iat-type \n')
        
        GenFeatInput.write(str(pm.Rcut)+','+str(pm.Rcut2)+','+str(pm.Rm)+','+str(pm.iflag_grid)+','+str(pm.fact_base)+','+\
            str(pm.dR1)+','+str(pm.dR2)+'      !Rc,Rc2,Rm,iflag_grid,fact_base,dR1,dR2 \n')
        GenFeatInput.write(str(pm.numOf2bfeat)+','+str(pm.numOf3bfeat1)+','+str(pm.numOf3bfeat2)+'       ! n2b, n3b1, n3b2 \n')
        GenFeatInput.write(str(pm.maxNeighborNum)+'      ! m_neigh \n')
        GenFeatInput.write(str(pm.E_tolerance)+'    ! E_tolerance  \n')
        GenFeatInput.write(str(pm.iflag_ftype)+'    ! iflag_ftype \n')
        GenFeatInput.write(str(pm.recalc_grid)+'    ! recalc_grid, 0 read from file, 1 recalc \n')



def writeFitInput():
    #fitInputPath=os.path.join(pm.fitModelDir,'fit.input')
    natom=200
    m_neigh=pm.maxNeighborNum
    n_image=200
    with open(pm.fitInputPath,'w') as fitInput:
        fitInput.write(str(pm.atomTypeNum)+', '+str(natom)+', '+str(m_neigh)+', '+\
                       str(n_image)+'      ! ntype,natom,m_neighb,nimage\n')
        for i in range(lem(pm.atomType)):
            line=str(pm.atomType[i])+', '+str(int(pm.fortranFitFeatNum0[i]))+', '+str(int(pm.fortranFitFeatNum2[i]))+\
                 ', '+str(int(pm.fortranGrrRefNum[i]))+', '+str(float(pm.fortranFitAtomRadii[i]))+', '+\
                 str(pm.fortranFitAtomRepulsingEnergies[i])+'       ! itype, nfeat0,nfeat2,ref_num,rad_atom,wp_atom\n'
            fitInput.write(line)
        fitInput.write(str(pm.fortranGrrKernelAlpha)+', '+str(pm.fortranGrrKernalDist0)+'            ! alpha,dist0 (for kernel)\n')
        fitInput.write(str(pm.fortranFitWeightOfEnergy)+', '+str(pm.fortranFitWeightOfEtot)+', '+str(pm.fortranFitWeightOfForce)+\
                       ', '+str(pm.fortranFitRidgePenaltyTerm)+','+str(pm.fortranFitCrossRad)+','+str(pm.fortranFitPower)+'        ! E_weight ,Etot_weight, F_weight, delta,rad3,power\n')
                       

def readFeatnum(infoPath):
    with open(infoPath,'r') as sourceFile:
        pm.realFeatNum=int(sourceFile.readline().split()[0])
            # pm.fortranFitFeatNum2[i]=pm.fortranFitFeatNum0[i]
    pm.fortranFitFeatNum0=pm.realFeatNum*np.ones((pm.atomTypeNum,),np.int32)
    pm.fortranFitFeatNum2=(pm.fortranFitFeatNum0*1.0).astype(np.int32)
    
def calFeatGrid():
    '''
    首先应该从设置文件中读取所有的用户设定

    Determine parameters:
    ---------------------
    pm.mulFactorVectOf2bFeat:    一维pm.mulNumOf2bFeat长度的cp.array,用于计算pm.mulNumOf2bFeat个二体feat的相应参数
    pm.mulFactorVectOf3bFeat:    一维pm.mulNumOf3bFeat长度的cp.array,用于计算pm.mulNumOf3bFeat个三体feat的相应参数
    pm.weightOfDistanceScaler:   标量实数，basic函数中对输入距离矩阵进行Scaler的权重w
    pm.biasOfDistanceScaler：    标量实数，basic函数中对输入距离矩阵进行Scaler的偏置b 
    '''    
    pm.mulFactorVectOf2bFeat=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.numOf2bfeat+2)[1:-1]-5.0)/4.0 +1)*pm.Rcut/2.0
    pm.mulFactorVectOf3bFeat1=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.numOf3bfeat1+2)[1:-1]-5.0)/4.0 +1)*pm.Rcut/2.0
    pm.mulFactorVectOf3bFeat2=((cp.logspace(cp.log10(1.0),cp.log10(9.0),pm.numOf3bfeat2+2)[1:-1]-5.0)/4.0 +1)*pm.Rcut2/2.0
    h2b=(pm.Rcut-float(pm.mulFactorVectOf2bFeat.max()))
    h3b1=(pm.Rcut-float(pm.mulFactorVectOf3bFeat1.max()))
    h3b2=(pm.Rcut2-float(pm.mulFactorVectOf3bFeat2.max()))

    with open(os.path.join(pm.OutputPath,'grid2.type2'),'w') as f:
        f.write(str(pm.numOf2bfeat)+' \n')
        for i in range(pm.numOf2bfeat):
            left=pm.mulFactorVectOf2bFeat[i]-h2b
            right=pm.mulFactorVectOf2bFeat[i]+h2b
            f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')
    with open(os.path.join(pm.OutputPath,'grid31.type2'),'w') as f:
        f.write(str(pm.numOf3bfeat1)+' \n')
        for i in range(pm.numOf3bfeat1):
            left=pm.mulFactorVectOf3bFeat1[i]-h3b1
            right=pm.mulFactorVectOf3bFeat1[i]+h3b1
            f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')
    with open(os.path.join(pm.OutputPath,'grid32.type2'),'w') as f:
        f.write(str(pm.numOf3bfeat2)+' \n')
        for i in range(pm.numOf3bfeat2):
            left=pm.mulFactorVectOf3bFeat2[i]-h3b2
            right=pm.mulFactorVectOf3bFeat2[i]+h3b2
            f.write(str(i)+'  '+str(left)+'  '+str(right)+' \n')


# def calGrid4()

if __name__ == "__main__":
    pass
