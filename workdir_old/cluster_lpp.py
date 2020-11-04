import pandas as pd
import numpy as np
import parameters as pm
import os
import prepare as pp
import feat_LPP as flpp
# import matplotlib.pyplot as plt
# %matplotlib inline


import copy

def read_data(filename,nfeat):

    df=pd.read_csv(filename,header=None,names=None)
    df=df.rename({1:'Type'},axis=1)
    # df=df.dropna()
    df.iloc[:][nfeat+2]=1.0
    return df

def readCluster(ClusterFile,):
    cluster=pd.read_csv(ClusterFile,delim_whitespace=True,header=None,names=None)
    cluster=cluster.rename({0:'Type'},axis=1)
    cluster=cluster.rename({1:'ClusterAssign'},axis=1)
    cluster=cluster.rename({2:'trainDataIndex'},axis=1)
    return cluster

def read_lpp(filename,nfeat2):
    df=pd.read_csv(filename,header=None,names=None)
    return df

def kmean_center(df,k,nfeat2):
    # df=read_data()
    CaseNum=len(df)
    centroids={
        i+1:[
            df.iloc[np.random.randint(0,CaseNum)][m] 
            for m in range(nfeat2)
            ]
        for i in range(k)
    }
    return centroids

def kmean_center_design(df,k,nfeat2,clusterDf):
    df=pd.concat([df,clusterDf], axis=1)
    # df=read_data()
    # CaseNum=len(df)
    centroids={
        i+1:[
            np.mean(df[df['ClusterAssign'] == i+1][m])
            for m in range(nfeat2)
            ]
        for i in range(k)
    }
    return centroids

def assignment(df,centroids,nfeat2):
    for i in centroids.keys():
        df['dist_from_{}'.format(i)]=0
        for m in range(nfeat2):
            df['dist_from_{}'.format(i)]=df['dist_from_{}'.format(i)]+(df.iloc[:][m]-centroids[i][m])**2
        df['dist_from_{}'.format(i)]=np.sqrt(df['dist_from_{}'.format(i)])
    
    centroids_dist_cols=['dist_from_{}'.format(i) for i in centroids.keys()]
    df['closest']=df.loc[:,centroids_dist_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('dist_from_')))
    return df

def update(centroids,df,nfeat2):
    for i in centroids.keys():
        for m in range(nfeat2):
            centroids[i][m] = np.mean(df[df['closest'] == i][m])
    return centroids    

def center_feat(df,k,nfeat):
    centroids={
        i+1:[
            np.mean(df[df['closest'] == i+1][m+3]) 
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids

def writefile(df,df2file):
    df.to_csv(df2file,index=False)

def saveType(df1,itype):
    df1=df1[df1['Type']==itype]
    df1.reset_index(inplace=True)
    df1=df1.drop('index',axis=1)
    return df1


def shiftFeat(df1,nfeat,shiftScale):
    for m in range(nfeat):
        df1.iloc[:][m+3]=(df1.iloc[:][m+3]-shiftScale.iloc[m][0])*shiftScale.iloc[m][1]
    return df1

def readShift(shiftfile):
    shiftScale=pd.read_csv(shiftfile,delim_whitespace=True,header=None)
    print(shiftScale.head())
    return shiftScale

def measureKmean(df,centroids,nfeat2):
    dist=0
    for i in centroids.keys():
        for m in range(nfeat2):
            dist=dist+np.sum((df[df['closest'] == i][m]-centroids[i][m])**2)
    return dist

    

# if __name__ == '__main__':
def runCluster(itype,alpha0=pm.alpha0,DesignCenter=pm.DesignCenter,k_dist0=pm.k_dist0,kernel_type=pm.kernel_type):

    nfeat=pm.realFeatNum
    nfeat2=pm.lppNewNum
    ClusterNum=pm.ClusterNum[itype-1]

# ********** input file   ***************  
    filename=os.path.join(pm.trainSetDir,'lppData.txt')
    ClusterFile=os.path.join('./output/','cluster_index.'+str(pm.atomType[itype-1]))
    
    lppFile=os.path.join('./output/','LPP')
    # centerfile='center.'+str(itype)
    file_w_feat='weight_feat.'+str(itype)
    shiftfile='feat_shift.'+str(itype)
    # centerfile=os.path.join(pm.fitModelDir,centerfile)
    file_w_feat=os.path.join(pm.fitModelDir,file_w_feat)
    shiftfile=os.path.join(pm.fitModelDir,shiftfile)
    weight_feat=np.loadtxt(file_w_feat)
    w_feat=weight_feat[:,1]**2

#*********** output file ************
    file_output='feat_cent.'+str(itype)
    # ClusterCase=['ClusterCase'+str(i+1) for i in range(ClusterNum)]
    df2file='df'+str(itype)+'.csv'
    file_output=os.path.join(pm.fitModelDir,file_output)
    df2file=os.path.join(pm.fitModelDir,df2file)
#*********** shift the feature ************
    shiftScale=readShift(shiftfile)
    df=read_data(filename,nfeat)
    # print(df.head())
    df=shiftFeat(df,nfeat,shiftScale)
    # print(df.head())
    # print(len(df))

#************* drop the other type ****************
    df=saveType(df,pm.atomType[itype-1])
    print(df.head())
#******************************************
    lpp_array=flpp.feat_lpp(itype,nfeat2,df.to_numpy())
    #for i in range(nfeat2):
    #    if (np.max(lpp_array[:,i])-np.min(lpp_array[:,i])) !=0.0:
    #        lpp_array[:,i]=lpp_array[:,i]/(np.max(lpp_array[:,i])-np.min(lpp_array[:,i])) 
        
    lppDf=pd.DataFrame(data=lpp_array)
   
#************ Design Center ****************
    centroids=kmean_center(lppDf,ClusterNum,nfeat2)    
#    print(centroids)
#************* k-mean cluster *****************
    lppDf=assignment(lppDf,centroids,nfeat2)
    centroids=update(centroids,lppDf,nfeat2)

    flg=0
    while True:
        flg=flg+1
        closest_centroids= lppDf['closest'].copy(deep=True)
        centroids=update(centroids,lppDf,nfeat2)
        lppDf=assignment(lppDf,centroids,nfeat2)
        if closest_centroids.equals(lppDf['closest']):
            break

    print(flg)
    print(lppDf.head())
    writefile(lppDf,df2file)
#*************** measure kmean ******************
    with open('./output/measureKmean','a') as f:
        dist=measureKmean(lppDf,centroids,nfeat2)
        f.write(str(dist)+' \n')

#*************** calc weight ********************
    df=pd.concat([df,lppDf], axis=1)  
    print(df.head())
    centroids_feat=center_feat(df,ClusterNum,nfeat)
    #print(centroids_feat)

    width={
        i+1:0
        for i in range(ClusterNum)
    }
    for i in centroids_feat.keys():
        for m in range(nfeat):       
            width[i]=width[i]+np.mean((df[df['closest']==i][m+3]-centroids_feat[i][m])**2*w_feat[m])

    # print(ClusterNum)
    with open(file_output,'w') as f:
        f.writelines(str(ClusterNum)+','+str(alpha0)+','+str(k_dist0)+','+str(int(kernel_type))+'\n')
        for i in centroids_feat.keys():
            for m in range(nfeat):
                f.write(str(centroids_feat[i][m]))
                f.write('  ')
            f.write('\n')
        for i in centroids_feat.keys():
            f.writelines(str(width[i])+'\n')
    # print(ClusterNum)

if __name__ == '__main__':
    shift=False    
    if shift:
        pp.collectAllSourceFiles()
        pp.readFeatnum(os.path.join(pm.sourceFileList[0],'info.txt'))
        import fortran_fitting as ff
        ff.makeFitDirAndCopySomeFiles()
        # readFittingParameters()
        ff.copyData()
        ff.writeFitInput()
        command='make pca -C'+pm.fitModelDir
        # print(command)
        os.system(command)
    itype=input('input itype:\n')
    runCluster(int(itype))


            

