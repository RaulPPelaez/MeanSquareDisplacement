


OPTIONS=-DUSE_GPU -DUSE_CPU

LIBRARIES= -lcufft -lfftw3 -lfftw3f

OPTIM=-O3
CXX=nvcc -expt-extended-lambda -x cu
#CXX=clang++

all: gitversion.h
	$(CXX) $(OPTIM) -std=c++11 MeanSquareDisplacement.cpp $(OPTIONS)   $(LIBRARIES) -o msd
gitversion.h: ../.git/HEAD ../.git/index
	echo "#ifndef GITVERSION" > $@
	echo "#define GITVERSION \"$(shell git rev-parse HEAD)\"" >> $@
	echo "#endif" >> $@

clean:
	rm -f gitversion.h msd
