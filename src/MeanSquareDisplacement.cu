/*Raul P. Pelaez 2018. Mean Square Displacement
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
 
    USAGE:
    
    $ cat file | msd -N X -Nsnapshots X > file.msd
    
    INPUT FORMAT:
    
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

      
      
    

    OUTPUT FORMAT:
    
    The output will have the following format:
    
    lag MSD_x MSD_y MSD_z

    



References:
[1] https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf
*/

#include"superRead.h"
#include"parseArguments.h"
#include<iostream>
#include"correlationGPU.cuh"
#include<vector>

using namespace FastCorrelation;
void print_help();

int main(int argc, char *argv[]){


  //cuFFT does not handle very well long signals in single precision, it is pretty lossy. 
  using real = double;
  
  //Parse command line options
  if(checkFlag(argc, argv, "-h") or checkFlag(argc, argv, "-help")){print_help(); exit(1);}
  
  int numberElements;   if(!parseArgument(argc, argv, "-Nsnapshots", &numberElements)){print_help();exit(1);}
  int nsignals = 1;     parseArgument(argc, argv, "-N", &nsignals);
  int windowSize = numberElements;    //parseArgument(argc, argv, "-windowSize", &windowSize);
  int maxLag = windowSize;
  // if(parseArgument(argc, argv, "-maxLag", &maxLag)){
  //   if(maxLag>windowSize){
  //     std::cerr<<"WARNING!: You should not ask for a lag time larger than the window size!"<<std::endl;
  //   }
  // }

    
  //I interpret time windows as just more signals, and store them as such.
  //So a single signal with 2 time windows is taken as 2 signals with half the time of the original signal.  
  int numberElementsReal = numberElements;
  int nsignalsReal = nsignals;
  int nWindows = numberElements/windowSize;
  numberElements = windowSize;
  nsignals *= nWindows;


  //This bad boys encode info of all particles, at all times in the three coordinates as:
  // x_i1_t1, y_i1_t1... x_i2_t1.. z_iN_t1, x_i1_t2 ... z_iN_tT
  
  std::vector<real> SAABB(3*numberElements*nsignals, 0);
  
  std::vector<float> r2(3*numberElements*nsignals);
  //Trajectories
  std::vector<cufftReal_t<real>> h_signalX(3*numberElements*nsignals);

  {
    //Read input
    int index = -1;
    for(int i = 0; i<numberElementsReal*nsignalsReal; i++){
      index++;
      double tmp[3];
      readNextLine(stdin, 3, tmp);

      //Compute SAABB(0) for each particle
      for(int k = 0; k<3; k++){
	r2[3*i+k] = tmp[k]*tmp[k];
	SAABB[3*(i%nsignals)+k] += 2.0*r2[3*i+k];
	h_signalX[3*i+k] = tmp[k];
      }
    }
    //Compute SAABB
    for(int i=1; i<numberElements; i++){
      for(int j= 0; j<nsignals; j++){
	for(int k=0; k<3; k++)
	  SAABB[3*(i*nsignals+j)+k] = SAABB[3*((i-1)*nsignals+j)+k] -
	    r2[3*((i-1)*nsignals+j)+k] -
	    r2[3*((numberElements-i)*nsignals+j)+k];
      }
    }

  }

  ScaleMode scaleMode = ScaleMode::none;
  
  real * S_AB = (real*)h_signalX.data();

  //The trajectories of all particles and coordinates are treated as just Nparticles*Ncoordinates independent signals.
  autocorrelationGPUFFT<real>(h_signalX.data(),
			      numberElements,
			      nsignals*3,
			      maxLag,
			      scaleMode);

  //Average the MSD for each particle at each time
  for(int i = 0; i<maxLag; i++){
    double msd[3] = {0,0,0};
    for(int k=0;k<3; k++){
      for(int s = 0; s<nsignals; s++){
	msd[k] += (SAABB[3*(i*nsignals + s)+k] - 2.0*S_AB[3*(i*nsignals+s)+k]);
      }
      msd[k] /= double(numberElements-i)*double(nsignals);
    }
    
    printf("%d %.8e %.8e %.8e\n", i, msd[0], msd[1], msd[2]);
  }




  return 0;
}
void print_help(){
  std::cerr<<"Raul P. Pelaez 2018. Mean Square Displacement"<<std::endl;
  std::cerr<<"  Computes the Mean Square Displacement of the trajectories saved in a superpunto-like file."<<std::endl;
  std::cerr<<"  The calculation is performed using cuFFT and the Fast Correlation Algorithm described in [1]."<<std::endl;
  std::cerr<<"  "<<std::endl;
  std::cerr<<"    MSD(dt) = < ( x_i(t0+dt) - x_i(t0) )^2 >_t0,i"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    MSD(dt) =  (1/(Np*Nt0)) sum_t0[ sum_i( (x_i(t0+dt)-x_i(t0))^2 )]"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    Defining:"<<std::endl;
  std::cerr<<"       S(dt) = sum_t0[  (x_i(t0+dt)-x_i(t0))^2 ]"<<std::endl;
  std::cerr<<"    It can be seen that:"<<std::endl;
  std::cerr<<"       S(dt) = SAABB(dt) - 2*SAB(dt)"<<std::endl;
  std::cerr<<"    Where:"<<std::endl;
  std::cerr<<"       SAABB(dt) = sum_t0( x_i(t0+dt)^2  + x_i(t0)^2)"<<std::endl;
  std::cerr<<"       SAB(dt) = sum_t0(x_i(t0)*x_i(t0+dt))"<<std::endl;
  std::cerr<<"      "<<std::endl;
  std::cerr<<"    SAABB has a pretty simple recursive relation:"<<std::endl;
  std::cerr<<"        SAABB(t0+dt) = SAABB(t0) - x_i(t0)^2 - x_i(T-t0)^2  -> T being the maximum lag"<<std::endl;
  std::cerr<<"	SAABB(0) = 2*sum_dt(x_i(dt)^2)"<<std::endl;
  std::cerr<<"       "<<std::endl;
  std::cerr<<"    And SAB(dt) is just the autocorrelation of each coordinate."<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    So MSD(dt) = <1/Nt0 ( SAABB(dt) - 2*SAB(dt) )>_particles"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    This code computes SAABB and SAB for each coordinate and particle, and then averages the resulting MSD for each particle."<<std::endl;
  std::cerr<<"    The correlation is computed using the discrete correlation theorem with cuFFT."<<std::endl;
  std::cerr<<" "<<std::endl;
  std::cerr<<"    USAGE:"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    $ cat file | msd -N X -Nsnapshots X > file.msd"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    INPUT FORMAT:"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    The input file must have three colums and each frame must be just after the previous one:"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    x_i1_t1 y_i1_t1 z_i1_t1"<<std::endl;
  std::cerr<<"    ..."<<std::endl;
  std::cerr<<"    x_iN_t1 ..."<<std::endl;
  std::cerr<<"    x_i1_t2 ..."<<std::endl;
  std::cerr<<"    ..."<<std::endl;
  std::cerr<<"    x_iN_tN ..."<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    You can go from superpunto format to this with:"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    $ cat spunto.pos | grep -v '#' > msdValidFile"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    If you have several realizations of a certain simulation and you want to average the resulting MSD you can just pass it all to msd interleaving the files and telling MSD that there are more particles. In other words:"<<std::endl;
  std::cerr<<"      If you want to compute the MSD of 10 simulations with 100 particles, you can interleave the frames and tell msd that you have 1000 particles."<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    You can interleave files and compute msd with this:"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    paste -d '\n' simulation*  | msd -N [whatever times the number of simulations] -Nsnapshots [whatever] > cool.msd"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"      "<<std::endl;
  std::cerr<<"      "<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    OUTPUT FORMAT:"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    The output will have the following format:"<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<"    lag MSD_x MSD_y MSD_z"<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"    "<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<""<<std::endl;
  std::cerr<<"References:"<<std::endl;
  std::cerr<<"[1] https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf"<<std::endl;
}
