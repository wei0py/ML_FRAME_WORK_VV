CALCFEAT=calc_gen_feature.f90 find_neighbore.f90 find_feature.f90 find_feature3.f90 find_feature4.f90 
CALCLINSRC=calc_lin_forceM_module.f90
CALCCLSTSRC=calc_lin_clusterM_module.f90
.PHONY: py


py: ../calc_feature.so ../calc_lin.so ../calc_clst.so

../calc_feature.so: $(CALCFEAT)
	python3 -m numpy.f2py -c -m calc_feature --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCFEAT)
	mv calc_feature.cpython-36m-x86_64-linux-gnu.so ../calc_feature.so

../calc_lin.so: $(CALCLINSRC)
	python3 -m numpy.f2py -c -m calc_lin --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCLINSRC)
	mv calc_lin.cpython-36m-x86_64-linux-gnu.so ../calc_lin.so

../calc_clst.so: $(CALCCLSTSRC)
	python3 -m numpy.f2py -c -m calc_clst --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCCLSTSRC)
	mv calc_clst.cpython-36m-x86_64-linux-gnu.so ../calc_clst.so

