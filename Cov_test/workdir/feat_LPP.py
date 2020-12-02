from lpproj import LocalityPreservingProjection 
import numpy as np
import parameters as pm
import os

def feat_lpp(itype,nfeat2,X):

    # X=np.loadtxt(os.path.join(pm.trainSetDir,'trainData.txt'),delimiter=',')
    X=X[:,4:-1]

    lpp = LocalityPreservingProjection(n_components=nfeat2,n_neighbors=pm.lpp_n_neighbors,weight=pm.lpp_weight,weight_width=pm.lpp_weight_width)
    """Locality Preserving Projection
 
    Parameters
    ----------
    n_components : integer
        number of coordinates for the manifold
 
    n_neighbors : integer
        number of neighbors to consider for each point.
 
    weight : string ['adjacency'|'heat']
        Weight function to use for the mapping
 
    weight_width : float
        Width of the heat kernel for building the weight matrix.
        Only referenced if weights == 'heat'
 
    neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
        Algorithm to use for nearest neighbors search,
        passed to neighbors.NearestNeighbors instance.
 
    Attributes
    ----------
    projection_ : array-like, shape (n_features, n_components)
        Linear projection matrix for the embedding

    n_components=2, 
    n_neighbors=5,              
    weight='adjacency', 
    weight_width=1.0,
    neighbors_algorithm='auto'
    """
    X_nD = lpp.fit_transform(X)

    np.savetxt('output/LPP.'+str(itype),X_nD)
    np.savetxt('output/eigv.'+str(itype),lpp.projection_)
    return X_nD
