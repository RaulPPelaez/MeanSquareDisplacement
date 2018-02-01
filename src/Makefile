

#You can add -DUSE_BOOST if you have boost installed to greatly speed up the reading process

all:
	nvcc -arch=sm_52  -std=c++11 -O3 MeanSquareDisplacement.cu -lcufft -o msd
