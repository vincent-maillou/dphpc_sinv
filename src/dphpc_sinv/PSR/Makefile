include ./make.inc
print-%: ; @echo $* = $($*)
INCLUDEDIR  = $(INCAZT) $(INCQHU) $(INCPGI) $(INCUFC) $(INCAMD) $(INCUMF) $(INCILS) \
		$(INCHYP) $(INCSLU) $(INCMPS) $(INCTDR) $(INCTHI) $(INCHDF5) $(INCMPI) $(INCGNU) $(INCMKL)
LIBS        = $(QHULL) $(SPARSEKIT) $(AZTEC) $(ARPACK) $(PARDISO) $(UMFPACK) $(ILS) \
		$(AMD) $(HYPRE) $(SUPERLU) $(MUMPS) $(PORD) $(METIS) $(TDR) $(THI) $(LIBSCI) \
		$(SCALAPACK) $(BLACS) $(LAPACK) $(BLAS) $(LINKS) $(OPENMP) $(MPI) $(F90_LIBS)

FILES  =  PSR.o


PSR: $(FILES) 
	$(LOADER) $(LOADFLAGS) $(FILES) $(LIBS) -lm -o $@_1

.C.o:
	$(MPICXX) -c $< $(CXXFLAGS) $(CXXDEFS) $(OPENMP) $(HAVE_SDEVICE) $(HAVE_ILS) $(QT_FLAGS) $(INCLUDEDIR)


clean:	
	rm -f *.o *.c *.h QTSolver
