import numpy as np
import pandas as pd
import os
import parameters as pm
import sys
# import pandas as pd
workpath=os.path.abspath(pm.codedir)
sys.path.append(workpath)
import prepare as pp




def write_natoms_dfeat():
    pp.collectAllSourceFiles()
    f_train_natom = open(pm.f_train_natoms,'w')
    f_test_natom = open(pm.f_test_natoms,'w')
    f_train_feat = open(pm.f_train_feat,'w')
    f_test_feat = open(pm.f_test_feat,'w')
    f_train_dfeat = open(pm.f_train_dfeat,'w')
    f_test_dfeat = open(pm.f_test_dfeat,'w')
    with open(os.path.join(pm.trainSetDir,'trainData.txt')) as f:
        get_all=f.readlines()
    
    dfeat_names = pd.read_csv(os.path.join(pm.trainSetDir,'inquirepos.txt'), header=None).values[:,1:].astype(int)
    # print(dfeat_names.shape)
    # print(dfeat_names[8,0])
    

    count=0
    Imgcount=0
    for system in pm.sourceFileList:
        
        infodata=np.loadtxt(os.path.join(system,'info.txt'))
        natom=infodata[1]
        ImgNum=infodata[2]-(len(infodata)-3)
        trainImgNum=int(ImgNum*(1-pm.test_ratio))
        trainImg=np.arange(0,trainImgNum)
        testImg=np.arange(trainImgNum,ImgNum)
        # with open(os.path.join(system,'info.txt'),'r') as sourceFile:
        #     sourceFile.readline()
        #     natom=int(sourceFile.readline().split()[0])
        #     ImgNum=int(sourceFile.readline().split()[0])
        # useImgs=np.arange(0,ImgNum,interval)
        
        for i in trainImg:
            f_train_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            # f_train_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(i+1)+'\n')
            f_train_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(dfeat_names[int(Imgcount+i),0])+', '+str(dfeat_names[int(Imgcount+i),1])+'\n')
            # for k,line in enumerate(get_all):
            #     if k >= count+i*natom and k < count+(i+1)*natom:             
            #         f_train_feat.writelines(line)        
            #     else:           
            #         continue  
        
        for i in testImg:
            # print(i)
            f_test_natom.writelines(str(int(natom))+' '+str(int(natom))+'\n')
            # f_test_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(i+1)+'\n')
            f_test_dfeat.writelines(str(os.path.join(system,'dfeat.fbin'))+', '+str(dfeat_names[int(Imgcount+i),0])+', '+str(dfeat_names[int(Imgcount+i),1])+'\n')
            # for k,line in enumerate(get_all):
            #     if k >= count+i*natom and k < count+(i+1)*natom:             
            #         f_test_feat.writelines(line)        
            #     else:           
            #         continue
        
        for k,line in enumerate(get_all):
            m=int((k-count)/natom)
            if m in trainImg:
                f_train_feat.writelines(line)
            if m in testImg:
                f_test_feat.writelines(line)   
            # if k >= count+i*natom and k < count+(i+1)*natom:             
                     
            # else:           
            #     continue

        count=count+natom*ImgNum
        Imgcount=Imgcount+ImgNum

    f_train_natom.close()
    f_test_natom.close()
    f_train_feat.close()
    f_test_feat.close()
    f_train_dfeat.close()
    f_test_dfeat.close()

if __name__=='__main__':
    if not os.path.isdir(pm.dir_work):
        os.system("mkdir " + pm.dir_work)
    for dirn in [pm.d_nnEi, pm.d_nnFi]:
        if not os.path.isdir(dirn):
            os.system("mkdir " + dirn)

    write_natoms_dfeat()