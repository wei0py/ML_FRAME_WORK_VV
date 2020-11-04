CALCFEAT=calc_gen_feature.f90 find_neighbore.f90 find_feature.f90 find_feature3.f90 find_feature4.f90 
CALCLINSRC=calc_lin_forceM_module.f90
CALCCLSTSRC=calc_lin_clusterM_module.f90
DFEAT=read_dfeatNN.f90
CALCFEATNN=calc_gen_feature_nn.f90 find_neighbore.f90 find_feature.f90 find_feature3.f90 find_feature4.f90 
.PHONY: py


py: ../calc_feature.so ../calc_lin.so ../calc_clst.so ../read_dfeatNN.so ../calc_feature_nn.so

../calc_feature.so: $(CALCFEAT)
	python3 -m numpy.f2py -c -m calc_feature --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCFEAT)
	mv calc_feature.cpython-36m-x86_64-linux-gnu.so ../calc_feature.so

../calc_lin.so: $(CALCLINSRC)
	python3 -m numpy.f2py -c -m calc_lin --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCLINSRC)
	mv calc_lin.cpython-36m-x86_64-linux-gnu.so ../calc_lin.so

../calc_clst.so: $(CALCCLSTSRC)
	python3 -m numpy.f2py -c -m calc_clst --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCCLSTSRC)
	mv calc_clst.cpython-36m-x86_64-linux-gnu.so ../calc_clst.so

../read_dfeatNN.so: $(DFEAT)
	python3 -m numpy.f2py -c -m read_dfeatNN --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(DFEAT)
	mv read_dfeatNN.cpython-36m-x86_64-linux-gnu.so ../read_dfeatNN.so

../calc_feature_nn.so: $(CALCFEATNN)
	python3 -m numpy.f2py -c -m calc_feature_nn --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCFEATNN)
	mv calc_feature_nn.cpython-36m-x86_64-linux-gnu.so ../calc_feature_nn.so
