#############################################################################
# Set the following options
#############################################################################

# Top directory
#HOME = /home/dleonard/Documents/library_path/
HOME = /home/dleonard/Documents/Year2/library_path/

# Platform
ifeq ($(BUILD_TYPE),)
#	BUILD_TYPE = ethz-intel64
#	BUILD_TYPE = ethz-gnu64-aps
	BUILD_TYPE = ethz-gnu64
endif

# Uncomment this to enable the coupling with D-Device
#HAVE_SDEVICE = -DSDEVICE

# Uncomment this to have shared memory capabilities for the lowest level of parallelism (only with intel compiler)
USE_SHARED_MEMORY = -DSHARED_MEMORY

HAVE_ILS = -DHAVE_ILS

#############################################################################
# Should not need to edit below, unless adding new platform
#############################################################################

# Default values which might be overridden
MAKE     = make
AR       = ar
ARFLAGS  = ruv
AR_FLAGS = ruv
RANLIB   = ranlib

# Platform/compiler specific definitions

ifeq ($(BUILD_TYPE),ethz-gnu64-aps)
        QT_FLAGS      = -DAdd_ -D__MINMAX -Wno-deprecated $(USE_SHARED_MEMORY) -Wno-write-strings
	GNU64	      = /usr/lib/gcc/x86_64-linux-gnu/8/
	MPIHOME	      = /usr/lib/x86_64-linux-gnu/openmpi/
	#INTEL64	      = /usr/pack/intel_compiler-2015.0.090-af/Linux-x86_64/
	#MKLHOME       = $(INTEL64)/mkl/lib/intel64
        #NANOTCAD      = /home/nanotcad/sw
        #GNU64         = $(NANOTCAD)/gcc/4.8.2/dahu
	#MPIHOME       = $(NANOTCAD)/mpich/3.1-gcc-4.8.2/dahu
        #INTEL64       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64
        MKLHOME       = $(HOME)/intel64
	BLAHOME	      = $(MKLHOME)
	LAPHOME       = $(MKLHOME)
	SCAHOME	      = $(MKLHOME)
        PARDISO_ARCH  = AMD64
        CC            = /usr/bin/gcc      -Wl,-rpath -Wl,$(GNU64)/lib
        F77           = /usr/bin/gfortran
        F90           = /usr/bin/gfortran
        GCC           = /usr/bin/gcc      -Wl,-rpath -Wl,$(GNU64)
        LOADER        = /usr/bin/mpic++ -Wl,-rpath -Wl,$(GNU64)
        MPIF90        = /usr/bin/mpif90 -Wl,-rpath -Wl,$(GNU64)
        MPICC         = /usr/bin/mpicc  -Wl,-rpath -Wl,$(GNU64)
        MPICXX        = /usr/bin/mpic++ -Wl,-rpath -Wl,$(GNU64)
	#NVCC	      = $(CUDAHOME)/bin/nvcc
        CCFLAGS       = -O3 -ffast-math -funroll-loops
        CC_FLAGS      = -O3 -ffast-math -funroll-loops 
        CXXFLAGS      = -O3 -ffast-math -funroll-loops -DMPICH_IGNORE_CXX_SEEK
        F77FLAGS      = -O3 -ffast-math -funroll-loops
        F77_FLAGS     = -O3 -ffast-math -funroll-loops
        F90FLAGS      = -O3 -ffast-math -funroll-loops
        F90_FLAGS     = -O3 -ffast-math -funroll-loops
        GCCFLAGS      = -O3 -ffast-math -funroll-loops
	#LOADFLAGS     = -O3 -ffast-math -funroll-loops -L$(GNU64) -lgfortran -L$(MPIHOME)/lib/ -lmpifort
        LOADFLAGS     = -O3 -ffast-math -funroll-loops -L$(GNU64) -lgfortran -L$(MPIHOME)/lib/ -lmpi -lmpi_mpifh
 	F90_LIBS      = -L$(GNU64) -lgfortran -Wl,-rpath,L$(GNU64)
	LAPACK        = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME)
	BLACS         = -L$(BLAHOME) -lmkl_blacs_sgimpt_lp64 -Wl,-rpath -Wl,$(MKLHOME)
	SCALAPACK     = -L$(SCAHOME) -lmkl_scalapack_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        LAPACK        = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME)
        #BLACS         = -L$(BLAHOME) -lmkl_blacs_sgimpt_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        #SCALAPACK     = -L$(SCAHOME) -lmkl_scalapack_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        LINKS         = -Wl,-rpath -Wl,$(GNU64)
        LIBSCI        = 
        BLAS          =
        OPENMP        = -fopenmp
	export LD_LIBRARY_PATH=$(GNU64)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
	export LD_LIBRARY_PATH=$(MKLHOME):$LD_LIBRARY_PATH
endif

ifeq ($(BUILD_TYPE),ethz-gnu64)
        QT_FLAGS      = -DAdd_ -D__MINMAX -Wno-deprecated $(USE_SHARED_MEMORY) -Wno-write-strings
        NANOTCAD      = /home/nanotcad/sw
	#GNU64	      = /usr/pack/gcc-8.2.0-af/linux-x64/
	GNU64	      = /usr/pack/gcc-11.2.0-af/linux-x64/
	#MPIHOME      = /usr/pack/mpich-3.2.1-af/linux-x64/
	MPIHOME	      = /usr/local/mpich-3.3.2/gcc/
	INTEL64	      = /usr/pack/intel_compiler-2015.0.090-af/Linux-x86_64/
	#MKLHOME       = $(INTEL64)/mkl/lib/intel64
	MKLHOME       = /home/dleonard/Documents/installs/miniconda3/envs/quatrex/lib/
        #NANOTCAD      = /home/nanotcad/sw
        #GNU64         = $(NANOTCAD)/gcc/4.8.2/dahu
	#MPIHOME       = $(NANOTCAD)/mpich/3.1-gcc-4.8.2/dahu
        #INTEL64       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64
        #MKLHOME       = $(INTEL64)/mkl/lib/em64t
        BLAHOME	      = $(MKLHOME)
        LAPHOME       = $(MKLHOME)
        SCAHOME	      = $(MKLHOME)
        PARDISO_ARCH  = AMD64
        CC            = $(GNU64)/bin/gcc      -Wl,-rpath -Wl,$(GNU64)/lib64
        F77           = $(GNU64)/bin/gfortran
        F90           = $(GNU64)/bin/gfortran
        GCC           = $(GNU64)/bin/gcc      -Wl,-rpath -Wl,$(GNU64)/lib64
        LOADER        = $(MPIHOME)/bin/mpic++ -Wl,-rpath -Wl,$(INTEL64)/lib
        MPIF90        = $(MPIHOME)/bin/mpif90 -Wl,-rpath -Wl,$(INTEL64)/lib
        MPICC         = $(MPIHOME)/bin/mpicc  -Wl,-rpath -Wl,$(INTEL64)/lib
        MPICXX        = $(MPIHOME)/bin/mpic++ -Wl,-rpath -Wl,$(INTEL64)/lib
	NVCC	      = $(CUDAHOME)/bin/nvcc
        CCFLAGS       = -O3 -ffast-math -funroll-loops
        CC_FLAGS      = -O3 -ffast-math -funroll-loops 
	#CXXFLAGS      = -O3 -ffast-math -funroll-loops -DMPICH_IGNORE_CXX_SEEK
	CXXFLAGS      = -O2 -funroll-loops -DMPICH_IGNORE_CXX_SEEK
        F77FLAGS      = -O3 -ffast-math -funroll-loops
        F77_FLAGS     = -O3 -ffast-math -funroll-loops
        F90FLAGS      = -O3 -ffast-math -funroll-loops
        F90_FLAGS     = -O3 -ffast-math -funroll-loops
        GCCFLAGS      = -O3 -ffast-math -funroll-loops
	#LOADFLAGS     = -O3 -ffast-math -funroll-loops -L$(GNU64)/lib64 -lgfortran -L$(INTEL64)/compiler/lib/intel64 -L$(MPIHOME)/lib/ -lmpifort
	LOADFLAGS     = -O2 -funroll-loops -L$(GNU64)/lib64 -lgfortran -L$(INTEL64)/compiler/lib/intel64 -L$(MPIHOME)/lib/ -lmpifort
 	F90_LIBS      = -L$(GNU64)/lib64 -lgfortran -Wl,-rpath,L$(GNU64)/lib64
	LAPACK        = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME)
	#BLACS         = -L$(BLAHOME) -lmkl_blacs_sgimpt_lp64 -Wl,-rpath -Wl,$(MKLHOME)
	#SCALAPACK     = -L$(SCAHOME) -lmkl_scalapack_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        #LAPACK        = -L$(LAPHOME) -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath -Wl,$(MKLHOME)
        #BLACS         = -L$(BLAHOME) -lmkl_blacs_sgimpt_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        #SCALAPACK     = -L$(SCAHOME) -lmkl_scalapack_lp64 -Wl,-rpath -Wl,$(MKLHOME)
        LINKS         = -Wl,-rpath -Wl,$(GNU64)/lib64
        LIBSCI        = 
        BLAS          =
        OPENMP        = -fopenmp
	export LD_LIBRARY_PATH=$(GNU64)/lib64:$(INTEL64)/lib
endif

ifeq ($(BUILD_TYPE),ethz-intel64)
	QT_FLAGS      = -DAdd_ -D__MINMAX -Wno-deprecated $(USE_SHARED_MEMORY)
	MPIHOME       = /home/mluisier/libraries/release/intel/mpich2/src/
	BLAHOME       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64/mkl/lib/em64t/
	SCAHOME       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64/mkl/lib/em64t/
	MKLHOME       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64/mkl/lib/em64t/
	INTHOME       = /usr/pack/intel_compiler-11.1.075-af/Linux-x86_64/lib/
	PARDISO_ARCH  = AMD64
	WSMP_ARCH     = X86_64/Intel
	WSMP_LIB      = wsmp64
	CC            = icc-11.1.075
	CXX           = icpc-11.1.075
	CPP	      = cpp
	MPICC         = /home/mluisier/libraries/release/intel/mpich2/bin/mpicc
	MPICXX        = /home/mluisier/libraries/release/intel/mpich2/bin/mpicxx
	GCC           = gcc
	F90           = ifort-11.1.075
	F77           = $(F90)
	MPIF90        = /home/mluisier/libraries/release/intel/mpich2/bin/mpif90
	LOADER        = $(MPICXX)
	GEN_FLAGS     =
	O_LEVEL       = -g
	GCCFLAGS      = -g
	CC_FLAGS      = -g $(GEN_FLAGS)
	CCFLAGS       = -g
	CPP_FLAGS     =	-g -P -traditional
	CXXFLAGS      = $(CCFLAGS) -DMPICH_IGNORE_CXX_SEEK
	F90FLAGS      = $(O_LEVEL) $(GEN_FLAGS)
	F77_FLAGS     = -g
	F77FLAGS      = $(F90FLAGS)
	LOADFLAGS     = $(CCFLAGS)
	F90_LIBS      = -L$(INTHOME) -lifcore -lifport -L$(INTHOME)/lsvml.a -Wl,-rpath,$(INTHOME)
	BLAS          =
        LAPACK        = -L$(MKLHOME) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,-rpath,$(MKLHOME)
	LINKS         = -lpthread -openmp -limf -lsvml -lm -lpthread
        BLACS         = -L$(BLAHOME) -lmkl_blacs_lp64
        SCALAPACK     = -L$(SCAHOME) -lmkl_scalapack_lp64
        OPENMP        = -openmp
endif

# Top library directory, relative to QTSolver
LIB_TOP       = $(HOME)

# Common include paths
INCAZT        = -I$(LIB_TOP)/AZTEC/lib/
INCMPI        = -I$(MPIHOME)/include/
INCMKL        = -I/home/dleonard/Documents/installs/miniconda3/envs/quatrex/include/

# Common library paths
ARPACKLIB     = $(LIB_TOP)/ARPACK/libarpack_SUN4.a
AMDLIB        = $(LIB_TOP)/AMD/Lib/libamd.a
AZTECLIB      = $(LIB_TOP)/AZTEC/lib/libaztec.a
METISLIB      = $(LIB_TOP)/Metis/libmetis.a
MUMPSLIB      = $(LIB_TOP)/MUMPS_4.10.0/lib/libzmumps.a
PORDLIB       = $(LIB_TOP)/MUMPS_4.10.0/lib/libpord.a
SUPERLULIB    = $(LIB_TOP)/SuperLU_DIST_2.0/Lib/libsuperlu_dist_2.0.a
UMFPACKLIB    = $(LIB_TOP)/UMFPACK/Lib/libumfpack.a

# Common link options
ARPACK        = -L$(LIB_TOP)/ARPACK/ -larpack_SUN4
AZTEC         = -L$(LIB_TOP)/AZTEC/lib/ -laztec
METIS         = -L$(LIB_TOP)/Metis/ -lmetis
MPI           = -L$(MPIHOME)/lib
PARDISO	      = -L$(LIB_TOP)/Pardiso/ -lpardiso
#PARDISO	      = $(LIB_TOP)/Pardiso/libpardiso500-GNU463-X86-64.so

ifneq ($(HAVE_SDEVICE),-DSDEVICE)
INCAMD        = -I$(LIB_TOP)/AMD/Include/
INCHYP        = -I$(LIB_TOP)/HYPRE/src/hypre/include/
INCMPS        = -I$(LIB_TOP)/MUMPS_4.10.0/include/
INCPOR        = -I$(LIB_TOP)/MUMPS_4.10.0/PORD/include/
INCSLU        = -I$(LIB_TOP)/SuperLU_DIST_2.0/SRC
INCUFC        = -I$(LIB_TOP)/UFconfig/
INCUMF        = -I$(LIB_TOP)/UMFPACK/Include

AMD           = -L$(LIB_TOP)/AMD/Lib/ -lamd
HYPRE	      = -L$(LIB_TOP)/HYPRE/src/hypre/lib -lHYPRE
MUMPS         = -L$(LIB_TOP)/MUMPS_4.10.0/lib/ -lzmumps -L$(LIB_TOP)/MUMPS_4.10.0/lib/ -lmumps_common
PORD          = -L$(LIB_TOP)/MUMPS_4.10.0/lib/ -lpord
SUPERLU       = -L$(LIB_TOP)/SuperLU_DIST_2.0/Lib/ -lsuperlu_dist_2.0
UMFPACK       = -L$(LIB_TOP)/UMFPACK/Lib/ -lumfpack
endif

# Common definitions
CD            = cd
ECHO          = echo
LN            = ln
LNFLAGS       = -s
RM            = rm
RMFLAGS       = -f
MV            = mv
MVFLAGS       = -f
SHELL         = sh
LIB           = -lm
LEX           = flex
YACC          = bison
SWIG	      = swig

ifeq ($(USE_SHARED_MEMORY),-DSHARED_MEMORY)
SMFLAG        = $(OPENMP)
endif
