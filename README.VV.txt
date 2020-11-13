VV is a model which goes beyond the linear model. 
The idea is that, it uses the origin input feature, generate the secondary features. 
The number of addition secondary feature is mm(itype), for each type, it can be different. 
Currently, we have used: exp(-feature), feature*feature, feature*feature*feature, to generate
secondary features. In the future, we can use more functions, or parameters. 
Thus, VV is a method between linear and NN. In NN, the nonlinear function has parameters W(i,j) 
to improve and adjust. Here, we use fixed pre-set nonlinear features. However, instead of using 
all these features (which can be rather large), we have used algorithms to select a subset (say
mm(type)) of these features, so there is no overfitting. In the future, we can improve the algorithms
to select this subset of features. 

To use VV, please have the following steps:

(step 1): run >python3 main.py, in parameter.py, specify isCalcFeat=True. 
          This will calculate the multiple feature types. 
          (Note, the main.py, and parameter.py are in this directory, same as for this README.txt, 
           but run should be done in a calculation directory, with sub-directories containing MOVEMENT files). 
(step 2): run >python3 main.py, in parameter.py, specify: isFitLinModel=True. 
          This first run the linear model. Actually, the main purpose is not the linear model itself, but
          some preprocess, like PCA analysis etc. In particular, it will generate a subdirectory: fread_dfeat. 
          Goto fread_dfeat, it will have feat_new_stored.1,2, etc. These are the combined feature, all store together, 
          as well as atomic energy Ei. They can be used to do analysis, and select the secondary VV feature (all based
          on the energy, not the force). 
(step 3): Goto fread_dfeat, run: select_mm_VV.r in the fit directory. One can run with scanning, which will scan the number
          of mm, to see which one has the minimum test set error. But in general, mm should be around 1000-2000. 
          Note, this should be done for each atom types. 
          Then, after scan, run select_mm_VV.r again, without scan, but with a input mm (e.g., from previous scan run). 
          This will generate OUT.VV_index.1,2, these are the files, specify which VV feature to be selected. 
(step 4): Stay in fread_dfeat directory, run: fit_VV_forceMM.r (from the fit subdirectory). It will generate 
          the linear_VV_fitB.ntype, this is the fitting parameter file (the model). Run: calc_VV_forceMM.r, it will 
          generate the predicted Ei and force, and compared with DFT results, with the files as: forceVV.pred.xx,
          energyVV.pred.xx, energyVV.pred.tot. Note, the energy, force weights are the same as in the linear 
          fitting results.  

-----------------------------------------------------
We still need to modify the calc_VV_forceMM.f, to make it a module, so we can run MD simulations. 
Right now, so so sure about the speed. Might need improvment later on. 
--------------------------------------------------
Note, the feature generation code is within: gen_feature
      the linear and VV fitting code is within: fit
      the MD code (modules) is within: workdir/fortran_code
-------------------------------------------------
         
        
