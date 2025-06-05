# Fast Mean Square Displacement (MSD)    
  
  This code computes the MSD of a list of trajectories using the FFT in O(N) time. It can be compiled/run with or without GPU (CUDA) support.
  
  The project exposes a CLI utility, a C++ library and a Python wrapper.
  
  
##  Installation:  

The project can be built and installed via CMake, or pip if the Python wrapper is needed.

Start by cloning the repository:

```bash
git clone https://github.com/RaulPPelaez/MeanSquareDisplacement
```
### Getting the dependencies  
Dependencies are handled with [conda](https://github.com/conda-forge/miniforge), and can be installed with the following command in the root of the repository:

```bash
conda env create
```

### Installing with CMake  
The following will compile the project and install it in the conda environment:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j$(nproc)
make install
```
After this, the `msd` command should be available in your terminal.
Tests for the C++ library can be run by running `ctest` in the build directory.

#### Additional CMake options  
- `USE_CUDA`: Compile in hybrid CPU/GPU mode, requires nvcc. Default is ON.
- `BUILD_TESTS`: Build the tests. Default is ON.
- `BUILD_EXECUTABLE`: Build the executable. Default is ON.

### Installing with pip  
If you want to use the Python wrapper, you can install it with pip from the root of the repository:

```bash
pip install .
```
This will install the Python package `mean_square_displacement`, which provides a Python interface to the C++ library.
You can test the correctness of the Python wrapper by running `pytest` in the root of the repository.

##  Usage:  
 
 The project allows you to compute the Mean Square Displacement (MSD) of a set of trajectories for a given set of particles and dimensions over time. If more than one dimension is provided, the MSD will be computed for each dimension separately. The output will be a file with the MSD values for each time step and dimension, averaged over all particles.
 
### CLI Utility:
``` bash  
  cat pos.dat | msd   
  -N [number of signals (particles)]  
  -Nsteps [number of lags in file]  
  -dimensions [=3, number of columns per signal to read]  
  -device [=auto, choose between cpu or gpu]   
  -precision [=double, can also be float. Float is much faster and needs half the memory]  
  > pos.msd  
   
```  
    
####  Input file format:    
The input file consists of a series of particle positions and dimensions at different time steps, which are separated by a line with `#` characters. The format is as follows:
```   
  #
  x1 y1 z1 ... //Coordinates of particle 1 at time 1  
  x2 y2 z2 ... //Coordinates of particle 2 at time 1  
  .  
  .  
  .  
  xN yN zN ...  
  #
  .  
  .  
  .  
  xN yN zN ... //coordinates of particle N at time Nsteps  
----  
```
####  Example:  
  ```
  ---pos.dat---  
  # #  
  0 0 0  
  0 0 0  
  # #  
  1 1 1  
  1 1 1  
  # #  
  2 2 2  
  2 2 2  
  ----  
  ```
  
```bash  
	cat pos.dat | msd -N 2 -Nsteps 3 -dimensions 3 > pos.msd  
```  
The output file `pos.msd` will contain the Mean Square Displacement for each time step and dimension, averaged over all particles. The first column will contain the time steps, and the subsequent columns will contain the MSD values for each dimension.

### Python Wrapper:
The following example generates a set of random walks in 2D and computes the MSD using the Python wrapper. Each dimension has a different diffusion coefficient.

```python
 from mean_square_displacement import mean_square_displacement
 # Test with different slopes for each dimension
 nparticles = 10000
 ndim = 2
 ntimes = 5000
 expected_slopes = [1, 2]  # Different slopes for each dimension
 
 positions = np.random.randn(nparticles, ndim, ntimes).astype(np.float64)
 positions[:, :, 0] = 0  # Start all particles at the origin
 positions = np.cumsum(positions, axis=2)  # Cumulative sum to simulate random walk
 for i in range(ndim):
     positions[:, i, :] *= np.sqrt(expected_slopes[i])
	 
 msd = mean_square_displacement(positions, "cpu")
 assert msd.shape == (ntimes, ndim)
 
 time = np.arange(ntimes)
 slopes = [
     fit_line(time, msd[:, i], discard_last_percentage=0.3) for i in range(ndim)
 ]
 
 assert np.allclose(
     slopes, expected_slopes, atol=0, rtol=0.05
 ), f"The slopes of the MSD should be close to 1 for Brownian motion. Got slopes: {slopes}"
```

### C++ API:
The C++ library exposes a single function:

```cpp

    
## References:    
[1] https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf    
    
    
