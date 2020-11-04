import pandas as pd
import numpy as np
import parameters as pm
import os
import prepare as pp
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


def kmean_center(df,k,nfeat):
    # df=read_data()
    CaseNum=len(df)
    centroids={
        i+1:[
            df.iloc[np.random.randint(0,CaseNum)][m+3] 
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids
def kmean_center_design1(df,k,nfeat,center):
    # df=read_data()
    # CaseNum=len(df)
    centroids={
        i+1:[
            df.iloc[center[i]][m+3] 
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids

def kmean_center_design(df,k,nfeat,clusterDf):
    df=pd.concat([df,clusterDf], axis=1)
    # df=read_data()
    # CaseNum=len(df)
    centroids={
        i+1:[
            np.mean(df[df['ClusterAssign'] == i+1][m+3])
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids

def assignment(df,centroids,nfeat,w_feat):
    for i in centroids.keys():
        df['dist_from_{}'.format(i)]=0
        for m in range(nfeat):
            df['dist_from_{}'.format(i)]=df['dist_from_{}'.format(i)]+(df.iloc[:][m+3]-centroids[i][m])**2*w_feat[m]
        df['dist_from_{}'.format(i)]=np.sqrt(df['dist_from_{}'.format(i)])
    
    centroids_dist_cols=['dist_from_{}'.format(i) for i in centroids.keys()]
    df['closest']=df.loc[:,centroids_dist_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('dist_from_')))
    return df

def update(centroids,df,nfeat):
    for i in centroids.keys():
        for m in range(nfeat):
            centroids[i][m] = np.mean(df[df['closest'] == i][m+3])
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

def measureKmean(df,centroids,nfeat,w_feat):
    dist=0
    for i in centroids.keys():
        for m in range(nfeat):
            dist=dist+np.sum((df[df['closest'] == i][m+3]-centroids[i][m])**2*w_feat[m])
    return dist

    

# if __name__ == '__main__':
def runCluster(itype,alpha0=pm.alpha0,DesignCenter=pm.DesignCenter,k_dist0=pm.k_dist0,kernel_type=pm.kernel_type):

    nfeat=pm.realFeatNum
    ClusterNum=pm.ClusterNum[itype-1]

# ********** input file   ***************  
    filename=os.path.join(pm.trainSetDir,'trainData.txt')
    ClusterFile=os.path.join('./output/','cluster_index.'+str(pm.atomType[itype-1]))
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
    # df=saveType(df,pm.atomType[itype-1])
    # print(df.head())
#************ Design Center ****************
    
    if DesignCenter:
        df=saveType(df,pm.atomType[itype-1])
        print(df.head())
        clusterDf=readCluster(ClusterFile)
        clusterDf=saveType(clusterDf,pm.atomType[itype-1])

        # center=np.loadtxt(centerfile,dtype=int)
        ClusterNum=np.max(clusterDf['ClusterAssign'])
        # print(center)
        print(ClusterNum)
        centroids=kmean_center_design(df,ClusterNum,nfeat,clusterDf)
    elif pm.DesignCenter2:
        center=pm.center[itype-1]
        ClusterNum=len(center)
        centroids=kmean_center_design1(df,ClusterNum,nfeat,center)
        df=saveType(df,pm.atomType[itype-1])
        print(df.head())
    else:
        df=saveType(df,pm.atomType[itype-1])
        print(df.head())
        centroids=kmean_center(df,ClusterNum,nfeat)
    
#    print(centroids)




#************* k-mean cluster *****************
    df=assignment(df,centroids,nfeat,w_feat)
    centroids=update(centroids,df,nfeat)

    flg=0
    while True:
        flg=flg+1
        closest_centroids= df['closest'].copy(deep=True)
        centroids=update(centroids,df,nfeat)
        df=assignment(df,centroids,nfeat,w_feat)
        if closest_centroids.equals(df['closest']):
            break
    
    print(flg)
    print(df.head())
#    writefile(df,df2file)
#*************** measure kmean ******************
    with open('./output/measureKmean','a') as f:
        dist=measureKmean(df,centroids,nfeat,w_feat)
        f.write(str(dist)+' \n')


#*************** calc weight ********************


    width={
        i+1:0
        for i in range(ClusterNum)
    }
    for i in centroids.keys():
        for m in range(nfeat):       
            width[i]=width[i]+np.mean((df[df['closest']==i][m+3]-centroids[i][m])**2*w_feat[m])

    # print(ClusterNum)
    with open(file_output,'w') as f:
        f.writelines(str(ClusterNum)+','+str(alpha0)+','+str(k_dist0)+','+str(int(kernel_type))+'\n')
        for i in centroids.keys():
            for m in range(nfeat):
                f.write(str(centroids[i][m]))
                f.write('  ')
            f.write('\n')
        for i in centroids.keys():
            f.writelines(str(width[i])+'\n')
    # print(ClusterNum)



if __name__ == '__main__':
    shift=True
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
    runCluster(2)


            

