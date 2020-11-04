# README

there is 3 folder in the MLcode, `fit`, `gen_feature` and `workdir`
you will need to edit the `parameters.py` in the `workdir` and run `python main.py` in `wordir`
*you need to copy `fit/fread_dfead.templ` as the `fitModelDir` you set in parameters.py.*


## workdir
1. edit the parameters.py
2. python main.py


there are 5 parts you need to pay attention. `Dir`, `for gen_feature.in`, `cluster input`,`for fit.input` and `for MD` 
And 4 main parameters you can choose to be true or false
```python
isCalcFeat=True
isFitLinModel=True
isClassify=True
isRunMd=True                                   #是否运行md
```
the `input` and `output` dir in `workdir` will store some input and output information, if you find something wrong, you may like to take a look at the files in them.
*each time you run `main.py`, the `gen_feature.in` and `location` in the `input` dir will update according to the lasted parameters.py.*


### isCalcFeat
if you set isCalcFeat= True, then the code will find the MOVEMENT file in `trainSetDir` and generate the features.
these are the parameters related to generate features
```fortran
   gen_feature.in
   2             !num of type 
   13            !atomic num of 1st type
   1             !atomic num of 2nd type
   5.5, 11.0, 5.0, 2, 0.2, 0.5, 0.5  !Rcut,Rcut2,Rm,iflag_grid 1 or 2 or 3,fact_base,dR1,dR2 
   40,5,5        !num of 2b feature, num of 3b feature, num of 3b feature2
   60            !max neighbor num
   0.3           !E tolerance, Etotp-Etotp_ave < E_tolerance
   3             ! iflag_ftype 2 or 3 or 4 when 4 iflag_grid must be 3
   1             ! recalc_grid, 0 read grid from file, 1 recalc grid
```


### isClassify
if you set isClassify=True, it will fitting the model using cluster method.
**IMPORTANT**
now the code use a smaller dataset lppData.txt to do classify. you can edit the interval in main.py if you want to adjust.
if you set use_lpp=True, then it will use LPP to help classification, DesignCenter is no use now. you need to set ClusterNum carefully.
you would better set a small lppNewNum in parameters.py and then use plotlpp.py to see if there are significant clusters. 
you can adjust the lpp parameters to make the clusters more obviously. Then set the ClusterNum by what you see.
this process may need to run the main.py for several times just set isClassify=True.

if you set DesignCenter=True, you need to edit workdir/input/classify.in.1 and classify.in.2 …(decide by how many types you have)
if you set DesignCenter=False, no need to edit, just set the ClusterNum you want
fitting results is in the `fitModelDir`: energyC.pred.* ,  forceC.pred.* ,  cluster_fitB.ntype

### isFitLinModel
if you set isFitLinModel=True, it will fitting the model using linear method.
fitting results is in the `fitModelDir`: energyL.pred.* ,  forceL.pred.* ,  linear_fitB.ntype

### isRunMd
if you set isRunMd=True, you need to edit the `for MD` part and decide the fitting model you want to use in the `fitModelDir`
in `mdImageFileDir`, you need to put the atom.config or MOVEMENT(using 1st image) you want to run md(#设置md的初始image的文件所在的文件夹 )



## parameters.py
```python
#************** Dir **********************
trainSetDir='../../AlHcomb_new/'                   # where you put the training set, should be dirs include MOVEMENT
fortranFitSourceDir='../fit'                       # where the fitting code is, it should be the fit dir in the MLcode
fitModelDir='../fit/fread_dfeat.comp'              # where your fitting results is
genFeatDir='../gen_feature/'                       # where the gen_feature code is
mdImageFileDir='../../MDtest'                      # 设置md的初始image的文件所在的文件夹

isCalcFeat=True                                # whether you want to generate feature when run main.py
isFitLinModel=True                             # whether you want to do linear fitting
isClassify=True                                 # whether you want to do cluster fitting
isRunMd=True                                   # 是否训练运行md  
isFollowMd=False                                #是否是接续上次的md继续运行 

#********* for gen_feature.in *********************
atomType=[1,13]                                 # the atomic numbers of the elements in your training system
maxNeighborNum=100                              # the max neighbor num that one atom could have
numOf2bfeat=24                                  # num of 2 body feature 
numOf3bfeat1=3                                  # num1 of 3 body feature, 相邻的
numOf3bfeat2=3                                  # num2 of 3 body feature, 对面的 

Rcut=5.5                                        # the distance around one atom that considered, for 2-body
Rcut2=5.5                                       # the distance around one atom that considered, for 3-body
Rm=5.0                                          # a parameter that influence feature generated when using iflag_grid = 1
iflag_grid=3                                    # 1 or 2 or 3, three types of grid that used for generate features, 3: read grid from grid2/31/32.type2(you can find them in wordir/output/)
fact_base=0.2                                   # a parameter that influence feature generated when using iflag_grid = 2
dR1=0.5                                         # a parameter that influence feature generated when using iflag_grid = 2, can be adjusted to optimize the features for fitting
dR2=0.5                                         # a parameter that influence feature generated when using iflag_grid = 2, can be adjusted to optimize the features for fitting
E_tolerance=0.3                    # using to choose images that Etotp-Etotp_ave < E_tolerance, only those images will be used
iflag_ftype=4                      # 2 or 3 or 4 when 4, iflag_grid must be 3; 4 now is same as old code

recalc_grid=1                      # 0:read grid from file; 1: recalc the grid
#----------------------------------------------------
rMin=0.0                          # no use
#************** cluster input **********************
kernel_type=2             # 1 is exp(-(dd/width)**alpha0), 2 is 1/(dd**alpha0+k_dist0**alpha0)
use_lpp=True
lppNewNum=3               # new feature num lpp generate. you can adjust more lpp parameters by editing feat_LPP.py. also see explains in it
lpp_n_neighbors=5         
lpp_weight='adjacency'    # 'adjacency' or 'heat'
lpp_weight_width=1.0
alpha0=1.0                # used in above functions
k_dist0=0.1                       # used in above functions
DesignCenter=False        # whether you want to use classification to decide the initial center for k-mean cluster
ClusterNum=[3,3]          # the cluster num you want for different atom type, should be in same order with atomType
#-----------------------------------------------

#******** for fit.input *******************************

fortranFitAtomRepulsingEnergies=[0.000,0.000]            #fortran fitting时对每种原子设置的排斥能量的大小，此值必须设置，无default值！(list_like)
fortranFitAtomRadii=[0.35,1.4]                        #fortran fitting时对每种原子设置的半径大小，此值必须设置，无default值！(list_like)
fortranFitWeightOfEnergy=0.9                    #fortran fitting时最后fit时各个原子能量所占的权重(linear和grr公用参数)  default:0.9
fortranFitWeightOfEtot=0.0                      #fortran fitting时最后fit时Image总能量所占的权重(linear和grr公用参数)  default:0.0
fortranFitWeightOfForce=0.1                     #fortran fitting时最后fit时各个原子所受力所占的权重(linear和grr公用参数)  default:0.1
fortranFitRidgePenaltyTerm=0.0001               #fortran fitting时最后岭回归时所加的对角penalty项的大小(linear和grr公用参数)  default:0.0001
#----------------------------------------------------

#*********************** for MD **********************

#以下部分为md设置的参数 
mdCalcModel='lin'                               #运行md时，计算energy和force所用的fitting model，‘lin' or 'clst'
mdRunModel='nve'                                #md运行时的模型,'nve' or 'nvt' or 'npt'
mdStepNum=1000                                  #md运行的步数
mdStepTime=0.2                                  #md运行时一步的时长(fs)
mdStartTemperature=300                          #md运行时的初始温度
mdEndTemperature=300                            #md运行采用'nvt'模型时，稳定温度
mdNvtTaut=0.1*1000                               #md运行采用'nvt'模型时，Berendsen温度对的时间常数

isTrajAppend=False                              #traj文件是否采用新文件还是接续上次的文件  default:False
isNewMovementAppend=False                       #md输出的movement文件是采用新文件还是接续上次的文件  default:False
mdTrajIntervalStepNum=1                         # md 每隔多少步输出到traj文件里
mdLogIntervalStepNum=1                          # md 每隔多少步输出到log文件里
mdNewMovementIntervalStepNum=1                  
mdStartImageIndex=0                             #若初始image文件为MOVEMENT,初始的image的编号  default:0

isOnTheFlyMd=False                              #是否采用on-the-fly md,暂时还不起作用  default:False
isFixedMaxNeighborNumForMd=False                #是否使用固定的maxNeighborNum值，默认为default,若为True，应设置mdMaxNeighborNum的值
mdMaxNeighborNum=None                           #采用固定的maxNeighborNum值时，所应该采用的maxNeighborNum值(目前此功能不可用)

isMdCheckVar=False                               #若采用 'grr' model时，是否计算var  default:False
isReDistribute=True                             #md运行时是否重新分配初速度，目前只是重新分配, default:True
velocityDistributionModel='MaxwellBoltzmann'    #md运行时,重新分配初速度的方案,目前只有'MaxwellBoltzmann',default:MaxwellBoltzmann

isMdProfile=False
```
