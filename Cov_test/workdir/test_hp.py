import parameters as pm 
import numpy as np


# import numpy as np
import random
import io
import os


inputfile="parameters.py"
outputfile="output/para_error"
# ref_num=[1500,1000]
fr=0.1
#dR1=[0.4,0.8,1.2,1.6,2.0]
#dR2=[0.4,0.8,1.2,1.6,2.0]
#dR1=[1.0,1.1,1.2,1.3,1.4]
#dR2=[1.4,1.5,1.6,1.7,1.8]
dR1=[]
dR2=[]
# alpha=[1]
# dist0=[1.45,1.55]
#os.system("nohup python main.py")
# os.system("../calc_lin_forceM.r")
with open(outputfile,'a+') as f2:
    # f2.write('reference num,force ratio: '+str(ref_num)+' '+str(fr)+'\n')
    f2.write('linear results ')
    filename="energyL.pred.1"
    filename=os.path.join(pm.fitModelDir,filename)
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)/len(data)
    f2.write(str(error)+' ')
    filename="energyL.pred.2"
    filename=os.path.join(pm.fitModelDir,filename)
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)/len(data)
    f2.write(str(error)+' ')
    filename="forceL.pred.1"
    filename=os.path.join(pm.fitModelDir,filename)
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)/len(data)
    f2.write(str(error)+' ')     
    filename="forceL.pred.2"
    filename=os.path.join(pm.fitModelDir,filename)
    data = np.loadtxt(filename)
    # plt.subplot(lenx,leny,k)
    error=np.sum((data[:,1]-data[:,0])**2,axis=0)/len(data)
    f2.write(str(error)+' '+'\n') 

with open(inputfile,'r') as f:
    get_all=f.readlines()

for s in dR1:
    for dis in dR2:
        with open(inputfile,'w') as f:
            for k,line in enumerate(get_all):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)                
                if 'dR1' in line:                              ## OVERWRITES line:1
                    f.writelines('dR1='+str(s)+" \n")    
                elif 'dR2' in line:
                    f.writelines('dR2='+str(dis)+" \n")   
                else:           
                    f.writelines(line)
        
        import parameters as pm
        os.system("nohup python main.py")
        # os.system("../calc_E_forceM.r")

        with open(outputfile,'a+') as f2:
            f2.write(str(s)+' '+str(dis)+' ')
            filename="energyL.pred.1"
            filename=os.path.join(pm.fitModelDir,filename)
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')
            filename="energyL.pred.2"
            filename=os.path.join(pm.fitModelDir,filename)
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')
            filename="forceL.pred.1"
            filename=os.path.join(pm.fitModelDir,filename)
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' ')     
            filename="forceL.pred.2"
            filename=os.path.join(pm.fitModelDir,filename)
            data = np.loadtxt(filename)
            # plt.subplot(lenx,leny,k)
            error=np.sum((data[:,1]-data[:,0])**2,axis=0)
            f2.write(str(error)+' '+'\n')                           




