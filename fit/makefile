ALLEXE = calc_lin_forceMM.r  feat_collect_PCA.r  fit_lin_forceMM.r
NORANEXE =  calc_lin_forceMM.r  feat_collect_PCA.r  fit_lin_forceMM.r
RANEXE = feat_collect_PCA.r
NORANOBJ = $(NORANEXE:.r=.o)
RANOBJ = $(RANEXE:.r=.o)
.PHONY: all lin 

all: $(ALLEXE)

	
$(NORANEXE): %.r : %.o
	ifort -o $@ $< -mkl
$(NORANOBJ): %.o : %.f
	ifort -c $<
$(RANEXE): %.r : %.o ran1.o
	ifort -o $@ $< ran1.o -mkl
$(RANOBJ): %.o : %.f
	ifort -c $<
ran1.o: ran1.f
	ifort -c ran1.f


lin: all
	make lin -Cfread_dfeat

