### Raul P. Pelaez 2019. Fast Mean Square Displacement    
  
      
  This code computes the MSD of a list of trajectories in spunto format using the FFT in O(N) time.  It can be compiled/run with or without GPU support.
  
##  COMPILE:  
	  
	Use cmake:  
```bash  
	mkdir build;  
	cd build;  
	cmake .. && make  
```  
  
##  USAGE:  
  
``` bash  
  cat pos.dat | msd   
  -N [number of signals (particles)]  
  -Nsteps [number of lags in file]  
  -dimensions [=3, number of columns per signal to read]  
  -device [=auto, choose between CUDA or FFTW: CPU or GPU]   
  -precision [=double, can also be float. Float is much faster and needs half the memory]  
  
  > pos.msd  
   
```  
    
##  FORMAT:    
  
```   
---pos.dat---  
  # #  
  x1 y1 z1 ... //Coordinates of particle 1 at time 1  
  x2 y2 z2 ...  
  .  
  .  
  .  
  xN yN zN ...  
  # #  
  .  
  .  
  .  
  xN yN zN ... //coordinates of particle N at time Nsteps  
----  
```
##  EXAMPLE:  
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
  
        
    
    
    
### References:    
[1] https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf    
    
    
