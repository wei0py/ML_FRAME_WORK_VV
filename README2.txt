   There are several characteristical of this new package

  (1) This directory only contains the source code, the user can copy the main.py and parameters.py 
      from the workdir subdirectory, to a truely working directory (We will call it A>). 
      change the:

      codedir='/ssd/linwang/ALL_ML_CODE/MLcode_MD.9.15.2020/workdir'
      fortranFitSourceDir='/ssd/linwang/ALL_ML_CODE/MLcode_MD.9.15.2020/fit'
      genFeatDir='/ssd/linwang/ALL_ML_CODE/MLcode_MD.9.15.2020/gen_feature'

      to the path name of the current path (of the absolute directory name of this README.txt)

      if you just untar this package, you might need to change the permision of the executables
      in workdir, fit, gen_feature, and test_code.
      E.g., go to one of these directories, do: 

      >chmod goa+x *.x *.r *.so *.py

      Within A>, one should put several subdirectories, each contains one MOVEMENT file 
      (the training data) 
      (We will call such subdirectory Ax1>, Ax2>, Ax3> ... directories. 

  (2) In parameters.py, set isCalcFeat=True (all other False)
      >conda activate YOURPROJECT_NAME     ! acivate the conda environment
      >python main.py

      This will generate the feature, 
      The isCalcFeat=True will generate: imagesNotused,  trainData.txt, MOVEMENTall, location, 
      and input output dir. 
      The real large file is in each Axj> subdirectorye, as dfeat.fbin (derivative of feature), and 
      info.txt. 
      imagesNotused: indicates which images are not used
      trainData.txt: has the Ei_case, and feature_case (so trainData.txt can be used for any testings, 
      only use the energy Ei_case, not the force, and no feature derivatives). 
      MOVEMENTall: all the movements used (-imagesNotused) addentated into a single file. 
      location: list the subdirectory path. 
      input: contains gen_feature.in (the input file used for the fortran code gen_feature.x). 
      outout: contains file for grid generation etc from gen_feature.x

 (3)  In parameters.py, set isFitLineModel=True, (all other False). 
      >python main.py     (Note, (2),(3) can be run at the same tie by setting these two flags to be T). 
   
      This will do a linear fitting (without PCA) for all the atom types, 
      and generate a subdirectory fread_dfeat
      In this subdirecotry "fread_dfeat", it has feat_new_stored.itype, energyL.pred.itype,
      forceL.pred.itype, linear_fitB.ntype (the fitted parameters), weight_feat.itype. 

      One can plot energyL.pred.itype, forceL.pred.itype to see the accuracy. 
      The shifted (center to 0) and  normalized feature for each type is now rewritted
      in feat_new_store.itype (copied from trainData.txt). The weights for each feature (through 
      multiple time linear fiting for a subset of the cases) are stored  in weight_feat.type
      This weight can be used as a metrix using this feature to measure the distance (very 
      ofen, we have required input "mm", to be used as: 
      dd(s1,s2)=sum_i(f1(i,s1)-f2(i,s2))**2*weight(i)^mm
      So mm=0 means do not use this metric  (in this case, the weight_feat.type is not requird). 

(4)   One can use the ">python main.py" to do other steps, e.g., cluster or MD, by changing 
      parameters.py. Here, we will introduce a set of fortran codes, to explore the other 
      fitting procedures etc. They are in the test_codes directory. These codes have not been 
      integrated in the python code, and they do not include the force at this moment. But we 
      like to test them for different cases, just to see how useful these procedures can be. 

      To test these codes, we suggest:
      (a) create a new subdirectory: TEST under the current A> directory. 
      (b) Copy location, feat_new_stored.itype, weight_feat.itype from fread_dfeat directory to 
          A/TEST> directory. 
      Then run the fortan codes in the test_codes directory (you can directly run like:
       /ssd/linwang/ALL_ML_CODE/MLcode_MD.9.15.2020/test_codes/AA_CODE.x 
      or have the patch in .bashrc, then directly run: AA_CODE.x. 
      Note, you might also like to copy LPP.input PCA.input to the A/TEST (but need to edit that), 
      and perhaps checkk  fread_dfeat/fit.input for the nfeat0 parameter

      Note, for the codes in test_codes, we have the following: 
      (a) feat_PCA.x: generate the PCA feature from trainData.txt, 
                      output 
                      feat_new.stored.PCA.itype (shift and normalized PCA features. For many subsequent 
                      calculation, if PCA featur, instead of original feature, like to be used, 
                      one needs to copy  feat_new.stored.PCA.itype  over to feat_new.stored.itype
                      weight_feat.PCA.itype to weight_feat.itype)

                      weight_feat.PCA.itype,feat_PV.type(PCA conversion feature vectors)

     (b) check_featdR.x: using feat_new.stored.itype and weight_feat.itype, to see when 
                      dd(s1,s2) is very close, how large is dE=|Ei_case(s1)-Ei_case(s2)|. 
                      If dE is large, while dd is close to zero, that means there is no way 
                      based on the current feature set, we can have a fitting better than dE. 
                      This gives us a sense of how accurate we can expect. 

    (c) LPP_test.x: a fortran code for LPP dimension reduction. It is written in Fortran so 
                    we can modify it. 

                    input: feat_new.stored.itype, weight_feat.itype 
                    output: LPP.itype, eigV.itype, LPP_eigen_feat.type
                    
                    The idea of LPP is to reduce the dimension of the feature, but make the 
                    connected points shrink closer, so it can more dramatically classify the cases. 

                    Suggest selections: imth=old; n_interval=20 (not to use every cases); 
                                        fact=6 (or 7), ikern=3(neigh), mm=0,1,2 (all okay, if 
                                        there is no weight_feat.type, mm=0)
                           
                    One can plot LLP.itype to see use the first two dimension (or other two 
                    dimensions) whether one can see cluster (using gnuplot). 
                    Note: in LPP.input, one needs to specify numVR (the number of reducted dimension), 
                    3, 4 are good selections). 

     (d) kmean_RD.x: input: LPP.itype, eigV.itype, and feat_new_stored.itype, 
                            do a kmean for all cases. 
                            It will ask for how many clusters. Perhaps 2-4 will be fine, depending on the physics. 
                    output: classf_cluster.type (the reduced dimension feature for clusters). 
                            classf_all.type (the reduced dimension feature for all cases). 
                            MOVEMENT.cluster (the MOVEMENT file, after been colored by classification 
                            for each atom (the iatom=iatom+id_cluster). 
                            Use convert_from_config.x < MOVEMENT.cluster, to get MOVEMENT.xsf. then 
                            show it in OVITO. Just to see whether the classification make physical sense. 

       Note: LPP_test.x, then kmean_RD.x should be used together. 
                       
    (e) classify_test.x: Another way (besides LPP_test.x/keam_RD.x) to do cluster (classification). 
                   See copy: test_codes/classify.in.example to A/TEST/classify.in. 
                   Edit classify.in for how to set the bond and angle. The idea is to use nearest 
                   neighbore information to do classification. It will do a kmean inside. 
                   output: classf_cluster.type
                           classf_all.type
                           MOVEMENT.cluster
                   Same as LPP_test.x/kmean_RD.x. 
                   Please use convert_from_config.x to plot out the MOVEMENT.cluster like above. 

    (f) linear_cluster.x: Do a cluster_linear fitting 
                  input:  feat_new_stored.type, classf_cluster.type, classf_all.type, 
                  output: It will first ask, whether to scan the parameter: factor
                          This is to optimize the super parameter for the kernel:

                          poly:   kern(s,c)=1/(d(s,c)**mm+(d_ave/fact)**mm)
                          exp :   kern(s,c)=exp(-(d(s,c)/(d_ave/fact))**mm)

                          here the distance d(s,c) between case s and cluster c is 
                          calculated based on the reduced dimension features
                          stored in classf_cluster.type(for c) and classf_all,type(for s)

                          These kernals are used as weights for case s, on different cluster c. 
                          These weights will be multiplied to feature in feat_new_stored.type, to 
                          generate new features (the number of new feature equals the number of 
                          cluster multiply the number of nfeat. 

                          The above fact can be scanned.  
                          The finite result (e.g., one can usef fact1=fact2=fact) is in E_fit_linear.2
                          one can plot it using gnuplot. 

  (g) linear_VV.x:  This is another approach, in contrast with the cluster approach, to go beyond 
                    simple linear fitting. 
                    For example, if use f(j,s) (the input feature from feat_new_stored.type) as input, j=1,nfeat 
                    then generates: 
                    f(new,s)= exp(-(f(j1,s)-fd(k1))^2/dw(k1)^2)*exp(-(f(j2,s)-fd(k2))^2/dw(k2)^2)
                    here: new=(j1,k1,j2,k2) the index of the new feature. 
                    or: 
                    f(new,s)=f(j1,s)*f(j2,s)*f(j3,s)
                    new=(j1,j2,j3). 

                    Note, as the number of the new featurs are extremly large. We have used a scheme to 
                    select the new features, depending on their dot-product to the residing dEi(s) from 
                    the linear fitting. We can select MM such new features. 
                    The code will ask whether to scan the MM. If yes, MM will be scan from 10 to 8000. 
                    We found that, usually 1000-2000 is good (so there will be 1000-2000 new features). 
                    During the scan, the code use 2/3 of the data as fitting, and 1/3 of the data as 
                    testing. Note, one should check the testing results. As MM getting bigger and bigger, 
                    the original fitting result will be smaller, but there is a optimal value for testing 
                    results.  

                    After the scan, one can run the code without scan, with a direct MM input (from 
                    previous run), then get E_fit.linear.type, E_fit.test.2 and plot them out. 
                    Note, the results on testing set, E_fit.test.2 is more important. 

                    Note: > "include feat**3, (0:no, 1:yes)", suggestion: select no (no f(j1)*f(j2)*f(j3)). 
                    Include it can be very slow, and it might not be good. One needs to test it. 

     Comment: in general, we find linear_VV.x and linear_cluster.x give similar quality results. 
                    linear_cluster.x is based on physical intuition, and linear_VV.x is based on 
                    brute force searching. We will test NN in the future. 
                          
                    
                  
                    
              
                     

    
                     
                                                



    
      
      
      
