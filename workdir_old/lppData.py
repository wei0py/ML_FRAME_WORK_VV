import parameters as pm
import numpy as np
import prepare as pp
import os

def read_data(inputfile):
    with open(inputfile,'r') as f:
        get_all=f.readlines()
    return get_all

def GenLppData(interval,lppDataFile):
    trainData=read_data(os.path.join(pm.trainSetDir,'trainData.txt'))
    pp.collectAllSourceFiles()
    with open(lppDataFile,'w') as lppF:
        count=0
        for system in pm.sourceFileList:
            infodata=np.loadtxt(os.path.join(system,'info.txt'))
            natom=infodata[1]
            ImgNum=infodata[2]-(len(infodata)-3)
            # with open(os.path.join(system,'info.txt'),'r') as sourceFile:
            #     sourceFile.readline()
            #     natom=int(sourceFile.readline().split()[0])
            #     ImgNum=int(sourceFile.readline().split()[0])
            useImgs=np.arange(0,ImgNum,interval)
            
            for img in useImgs:
                for k,line in enumerate(trainData):
                    if k >= count+img*natom and k < count+(img+1)*natom:             
                        lppF.writelines(line)        
                    else:           
                        continue  
            count=count+natom*ImgNum

if __name__=='__main__':
    lppDataFile=os.path.join(pm.trainSetDir,'lppData.txt')
    interval=50
    GenLppData(interval,lppDataFile)
