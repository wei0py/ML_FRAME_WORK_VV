CALCFTYPE1=gen_2b_feature.f90 find_neighbore00.f90 find_neighbore.f90 find_feature_2b_type1.f90 find_feature_2b_type2.f90 find_feature_2b_type3.f90
CALCFTYPE2=gen_3b_feature.f90 find_neighbore00.f90 find_neighbore.f90 find_feature_3b_type1.f90 find_feature_3b_type2.f90 find_feature_3b_type3.f90
CALCLINSRC=calc_lin_forceM_module.f90
CALCCLSTSRC=calc_lin_clusterM_module.f90
DFEAT=read_dfeatNN.f90

.PHONY: py


py: ../calc_ftype1.so ../calc_ftype2.so ../calc_lin.so ../calc_clst.so ../read_dfeatNN.so 

../calc_ftype1.so: $(CALCFTYPE1)
	python3 -m numpy.f2py -c -m calc_ftype1 --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCFTYPE1)
	mv calc_ftype1.cpython-36m-x86_64-linux-gnu.so ../calc_ftype1.so

../calc_ftype2.so: $(CALCFTYPE2)
	python3 -m numpy.f2py -c -m calc_ftype2 --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCFTYPE2)
	mv calc_ftype2.cpython-36m-x86_64-linux-gnu.so ../calc_ftype2.so

../calc_lin.so: $(CALCLINSRC)
	python3 -m numpy.f2py -c -m calc_lin --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCLINSRC)
	mv calc_lin.cpython-36m-x86_64-linux-gnu.so ../calc_lin.so

../calc_clst.so: $(CALCCLSTSRC)
	python3 -m numpy.f2py -c -m calc_clst --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(CALCCLSTSRC)
	mv calc_clst.cpython-36m-x86_64-linux-gnu.so ../calc_clst.so

../read_dfeatNN.so: $(DFEAT)
	python3 -m numpy.f2py -c -m read_dfeatNN --fcompiler=intelem --compiler=intelem -L$MKLROOT/lib/intel64/ -lmkl_rt $(DFEAT)
	mv read_dfeatNN.cpython-36m-x86_64-linux-gnu.so ../read_dfeatNN.so

