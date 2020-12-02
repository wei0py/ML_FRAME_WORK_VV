import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
import parameters as pm
import os
import prepare as pp

import copy

def read_data(filename,nfeat):
    # filename='newfeat'
    # data=np.loadtxt(filename)
    # data1=np.zeros((nfeat,ncase))
    lst=[]
    with open(filename,'r') as f:
        data=f.readlines()
        for line in data:
            odom=line.split()
            # numbers_float=map(float,odom)
            for k in range(len(odom)):
                lst.append(float(odom[k]))
    lst=np.asarray(lst)
    data1=np.reshape(lst,(-1,nfeat+1))

            # data1[:,k]=data
    # df=pd.read_csv(filename,delim_whitespace=True,header=None)
    df=pd.DataFrame(data1,index=[i for i in range(data1.shape[0])],columns=[i for i in range(data1.shape[1])])
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
            df.iloc[np.random.randint(0,CaseNum)][m+1] 
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids

# def kmean_center_design(df,k,nfeat,center):
#     # df=read_data()
#     CaseNum=len(df)
#     centroids={
#         i+1:[
#             df.iloc[center[i]][m+1] 
#             for m in range(nfeat)
#             ]
#         for i in range(k)
#     }
#     return centroids
def kmean_center_design(df,k,nfeat,clusterDf):
    # print(len(df))
    # print(len(clusterDf))
    # print(df.head())
    # print(clusterDf.head())
    df=pd.concat([df,clusterDf], axis=1)
    # print(df.head())
    # df=read_data()
    # CaseNum=len(df)
    centroids={
        i+1:[
            np.mean(df[df['ClusterAssign'] == i+1][m+1])
            for m in range(nfeat)
            ]
        for i in range(k)
    }
    return centroids

def assignment(df,centroids,nfeat,w_feat):
    for i in centroids.keys():
        df['dist_from_{}'.format(i)]=0
        for m in range(nfeat):
            df['dist_from_{}'.format(i)]=df['dist_from_{}'.format(i)]+(df.iloc[:][m+1]-centroids[i][m])**2*w_feat[m]
        df['dist_from_{}'.format(i)]=np.sqrt(df['dist_from_{}'.format(i)])
    
    centroids_dist_cols=['dist_from_{}'.format(i) for i in centroids.keys()]
    df['closest']=df.loc[:,centroids_dist_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('dist_from_')))
    return df

def update(centroids,df,nfeat):
    for i in centroids.keys():
        for m in range(nfeat):
            centroids[i][m] = np.mean(df[df['closest'] == i][m+1])
    return centroids    

def writefile(df,df2file):
    df.to_csv(df2file,index=False)

def saveType(df1,itype):
    df1=df1[df1['Type']==itype]
    df1.reset_index(inplace=True)
    df1=df1.drop('index',axis=1)
    return df1



if __name__ == '__main__':
    itype=1
    ClusterNum=4
    nfeat=80
    alpha0=1.0
    kmean_input='kmean_input'
    kmean_input=os.path.join(pm.fitModelDir,kmean_input)
    kmean_data=np.loadtxt(kmean_input)
    itype=int(kmean_data[0])
    ClusterNum=int(kmean_data[1])
    nfeat=int(kmean_data[2])
    alpha0=kmean_data[3]
    DesignCenter=int(kmean_data[4])
    filename='newfeat.'+str(itype)
    file_w_feat='weight_feat.'+str(itype)
    file_output='feat_cent.'+str(itype)
    # ClusterCase=['ClusterCase'+str(i+1) for i in range(ClusterNum)]
    df2file='df'+str(itype)+'.csv'
    centerfile='center.'+str(itype)
    filename=os.path.join(pm.fitModelDir,filename)
    file_output=os.path.join(pm.fitModelDir,file_output)
    df2file=os.path.join(pm.fitModelDir,df2file)
    file_w_feat=os.path.join(pm.fitModelDir,file_w_feat)
    ClusterFile=os.path.join('./output/','cluster_index')

    weight_feat=np.loadtxt(file_w_feat)
    w_feat=weight_feat[:,1]**2

    #ncase=12551

    df=read_data(filename,nfeat)
    # nfeat=len(df.iloc[0,:])-1
    # print(df.head())

    if DesignCenter==0:
        # center=np.loadtxt(centerfile,dtype=int)
        # centroids=kmean_center_design(df,ClusterNum,nfeat,center)
        clusterDf=readCluster(ClusterFile)
        # print(clusterDf.head())
        clusterDf=saveType(clusterDf,pm.atomType[itype-1])
        # print(clusterDf.head())

        # center=np.loadtxt(centerfile,dtype=int)
        ClusterNum=np.max(clusterDf['ClusterAssign'])
        # print(center)
        print(ClusterNum)
        centroids=kmean_center_design(df,ClusterNum,nfeat,clusterDf)
    else:
        centroids=kmean_center(df,ClusterNum,nfeat)

    # print(centroids)
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



    width={
        i+1:0
        for i in range(ClusterNum)
    }
    for i in centroids.keys():
        for m in range(nfeat):       
            width[i]=width[i]+np.mean((df[df['closest']==i][m+1]-centroids[i][m])**2*w_feat[m])

    with open(file_output,'w') as f:
        f.writelines(str(ClusterNum)+','+str(alpha0)+'\n')
        for i in centroids.keys():
            for m in range(nfeat):
                f.write(str(centroids[i][m]))
                f.write('  ')
            f.write('\n')
        for i in centroids.keys():
            f.writelines(str(width[i])+'\n')
    



            

