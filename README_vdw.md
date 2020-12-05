### several changes

1. adding vdw
    first gen feature, then set isFitVdw=True, run main.py. It will generate vdw_fitB.ntype in fitModelDir, which will be used in fitting and md process. If you do not want to use this term, just modify the vdw_fitB.ntype in fitModelDir as what you want. This file must exist when fitting and running md.
2. NN results will be in fitModelDir
