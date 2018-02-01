### Raul P. Pelaez 2018. Mean Square Displacement  

  Computes the Mean Square Displacement of the trajectories saved in a superpunto-like file.  
  The calculation is performed using cuFFT and the Fast Correlation Algorithm described in [1].  
    
    MSD(dt) = < ( x_i(t0+dt) - x_i(t0) )^2 >_t0,i  
      
    MSD(dt) =  (1/(Np*Nt0)) sum_t0[ sum_i( (x_i(t0+dt)-x_i(t0))^2 )]  
      
    Defining:  
       S(dt) = sum_t0[  (x_i(t0+dt)-x_i(t0))^2 ]  
    It can be seen that:  
       S(dt) = SAABB(dt) - 2*SAB(dt)  
    Where:  
       SAABB(dt) = sum_t0( x_i(t0+dt)^2  + x_i(t0)^2)  
       SAB(dt) = sum_t0(x_i(t0)*x_i(t0+dt))  
        
    SAABB has a pretty simple recursive relation:  
        SAABB(t0+dt) = SAABB(t0) - x_i(t0)^2 - x_i(T-t0)^2  -> T being the maximum lag  
	SAABB(0) = 2*sum_dt(x_i(dt)^2)  
         
    And SAB(dt) is just the autocorrelation of each coordinate.  
  
    So MSD(dt) = <1/Nt0 ( SAABB(dt) - 2*SAB(dt) )>_particles  
  
    This code computes SAABB and SAB for each coordinate and particle, and then averages the resulting MSD for each particle.  
    The correlation is computed using the discrete correlation theorem with cuFFT.  
   
##    USAGE:  
      
    $ cat file | msd -N X -Nsnapshots X > file.msd  
      
###     INPUT FORMAT:  
      
    The input file must have three colums and each frame must be just after the previous one:  
  
    x_i1_t1 y_i1_t1 z_i1_t1  
    ...  
    x_iN_t1 ...  
    x_i1_t2 ...  
    ...  
    x_iN_tN ...  
  
    You can go from superpunto format to this with:  
      
    $ cat spunto.pos | grep -v "#" > msdValidFile  
  
    If you have several realizations of a certain simulation and you want to average the resulting MSD you can just pass it all to msd interleaving the files and telling MSD that there are more particles. In other words:  
      If you want to compute the MSD of 10 simulations with 100 particles, you can interleave the frames and tell msd that you have 1000 particles.  
  
    You can interleave files and compute msd with this:  
  
    paste -d '\n' simulation*  | msd -N [whatever times the number of simulations] -Nsnapshots [whatever] > cool.msd  
  
        
        
      
  
###    OUTPUT FORMAT:  
      
    The output will have the following format:  
      
    lag MSD_x MSD_y MSD_z  
  
      
  
  
  
### References:  
[1] https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf  
  
  
