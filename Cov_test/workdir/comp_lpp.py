import pandas as pd
import numpy as np
import parameters as pm
import os
import prepare as pp


import copy

MOVEMENTlppFile=os.path.join(pm.trainSetDir,'MOVEMENTlpp.xyz')
with open(MOVEMENTlppFile) as f:
    get_all=f.readlines()

#names=locals()
dfcsv=[]
for itype in range(len(pm.atomType)):
    dfcsv.append(str(os.path.join(pm.fitModelDir,'df'+str(itype+1)+'.csv')))

df1=pd.read_csv(dfcsv[0])
df2=pd.read_csv(dfcsv[1])
# df1.set_index(['index'],inplace=True)
# df2.set_index(['index'],inplace=True)
    #print(names['df'+str(itype+1)].head())
print(df1.head())
print(df2.head())
df=pd.concat([df1,df2],axis=0,join='inner',sort=True)
df.sort_values('index',inplace=True)
df.to_csv('df',index=False)
print(df.head())
with open('newMOVEMENT.xyz','w') as newfile:
    # countlist=[]
    count=0
    for k,line in enumerate(get_all):
        if len(line.split())==1:
            natom=line.split()[0]
            count=count+1
            newfile.writelines(line)
        elif 'Iteration' in line:
            newfile.writelines(line)
        else:    
            newfile.writelines(str(df.iloc[k-count*2]['closest']+df.iloc[k-count*2]['Type'])+'  '+line)
        


            

        



