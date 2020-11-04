# need to edit
## main.py
- delete two trainData.
- write the input for gen 2b and 3b into ./input
- run gen 2b and gen 3b
- write FeatNum into ./input and import it to parameters

## workdir/prepare.py
- write gen 2b input function and 3b
- grid2b_type3.* , grid3b_b1b2_type3.* (n3b2) , grid3b_cb12_type3.* (n3b1)


## parameters.py



## Question

<!-- 1. Rc in 2b and 3b same or different? -->
<!-- 2. feature num is different now. many points should change -->
3. now only 2b and 3b, could exists other type? how to define the parameters?
4. to do md, do I need input all types of features? or just return features and combine them in python. Then input another fortran package


## Idea
1. in md, could return feature2b and feature3b, then feature[:, :nfeat2b]=feature2b, feature[:, nfeat2b:(nfeat2b+nfeat3b)]=nfeat3b
