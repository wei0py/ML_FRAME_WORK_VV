import os
import sys
import parameters as pm
workpath=os.path.abspath(pm.codedir)
sys.path.append(workpath)
import prepare as pp
import lppData as lppData

# genFeatInputFile='./gen_feature.in'
if os.path.exists('./input/'):
    pass
else:
    os.mkdir('./input/')
    os.mkdir('./output')



if pm.isCalcFeat:
    if os.path.exists(os.path.join(os.path.abspath(pm.trainSetDir),'trainData.txt')):
        os.system('rm '+os.path.join(os.path.abspath(pm.trainSetDir),'trainData.txt'))
    else:
        pass
    if os.path.exists(os.path.join(pm.trainSetDir,'lppData.txt')):
        os.system('rm '+os.path.join(pm.trainSetDir,'lppData.txt'))
    else:
        pass
    # os.system('rm '+os.path.join(os.path.abspath(pm.trainSetDir),'trainData.txt'))
    # os.system('rm '+os.path.join(os.path.abspath(pm.trainSetDir),'MOVEMENTall'))
    pp.collectAllSourceFiles()
    pp.savePath()
    pp.combineMovement()
    pp.writeGenFeatInput()
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    if pm.iflag_grid == 3:
        pp.calFeatGrid()
    command=pm.genFeatDir+"/gen_feature.x > ./output/out"
    os.system(command)
    pp.movementUsed()
else:
    os.system('cp '+os.path.abspath(pm.fbinListPath)+' ./input/')
    pp.writeGenFeatInput()
    pp.collectAllSourceFiles()
    pp.movementUsed()

if pm.isClassify:
    if os.path.exists(os.path.join(pm.trainSetDir,'lppData.txt')):
        pass
    else:
        interval=50
        lppData.GenLppData(interval,os.path.join(pm.trainSetDir,'lppData.txt'))

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
    if pm.use_lpp:
        import cluster_lpp as clst
    else:
        import cluster_trainData as clst
        if pm.DesignCenter:
            for i in range(pm.atomTypeNum):
                os.system('cp '+os.path.join(pm.InputPath,'classify.in.'+str(i+1))+' '+os.path.join(pm.InputPath,'classify.in'))
                os.system(pm.genFeatDir+'/classification.x')
    for i in range(pm.atomTypeNum):
        clst.runCluster(i+1)
    command='make clst -C'+pm.fitModelDir
    # print(command)
    os.system(command)
    

#os.system('python cluster_trainData.py')

if pm.isFitLinModel:
    import fortran_fitting as fortranfit
    pp.readFeatnum(os.path.join(pm.sourceFileList[0],'info.txt'))
    fortranfit.fit()

if pm.isRunMd:
    # import preparatory_work as ppw
    from md_runner import MdRunner
    # if not pm.isCalcFeat:
    #     ppw.preparatoryWorkForMdImage()
    mdRunner=MdRunner()
    for i in range(pm.mdStepNum):
        mdRunner.runStep()
    mdRunner.final()

