all: classify_test.x LPP_test.x kmean_RD.x linear_cluster.x check_featdE.x linear_VV.x feat_PCA.x

feat_PCA.x: 
	ifort feat_PCA.f ran1.f -mkl \
		-o $@

check_featdE.x: 
	ifort check_featdE.f -mkl \
		-o $@

LPP_test.x: 
	ifort LPP_test.f -mkl \
		-o $@

kmean_RD.x: 
	ifort scan_title.f90 transform_to_upper.f90 ran1.f kmean_RD.f90 -mkl \
		-o $@

classify_test.x: 
	ifort mod_bond_angle.f90 scan_title.f90 transform_to_upper.f90 find_neighbore.f90 classify3.f90 kmean.f90 \
	classify_test.f90 \
		-o $@

linear_cluster.x: 
	ifort linear_cluster.f -mkl \
		-o $@

linear_VV.x: 
	ifort linear_VV.f ran1.f -mkl \
		-o $@


clean:
	rm -f *.x *.mod
