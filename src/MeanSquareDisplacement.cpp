/*Raul P. Pelaez 2019. Fast Mean Square Displacement
  
  This code computes the MSD of a list of trajectories in spunto format using the FFT in O(N) time.


  USAGE:

  cat pos.dat | msd 
  -N [number of signals (particles)]
  -Nsteps [number of lags in file]
  -dimensions [=3, number of columns per signal to read]
  -device [=auto, choose between CUDA or FFTW: CPU or GPU] 
  -precision [=double, can also be float. Float is much faster and needs half the memory]

  > pos.msd
  
  FORMAT:

  ---pos.dat---
  #
  x1 y1 z1 ... //Coordinates of particle 1 at time 1
  x2 y2 z2 ...
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
  EXAMPLE:

  ---pos.dat---
  #
  0 0 0
  0 0 0
  #
  1 1 1
  1 1 1
  #
  2 2 2
  2 2 2
  ----

  cat pos.dat | msd -N 2 -Nsteps 3 > pos.msd



 */


#include<iostream>
#include<algorithm>
#include<numeric>
#include<fstream>
#include<cstring>
#include<unistd.h>
#include"gitversion.h"
#include"defines.h"
#include"superIO.h"

#include"autocorrGPU.cuh"
#include"autocorrCPU.h"

namespace MeanSquareDisplacement{ 
  enum class device{
    gpu, cpu, none
  };

  template<class real, class ...T>
  std::vector<real> autocorr(device dev, T... args){
    if(dev == device::cpu)
      return autocorrCPU<real>(args...);    
    else if(dev == device::gpu)
      return autocorrGPU<real>(args...);
    else{
      std::cerr<<"UNKNOWN DEVICE!"<<std::endl;
      exit(1);
    }
      
  }
#ifdef USE_GPU
  bool gpu_mode_available = true;
#else
  bool gpu_mode_available = false;
#endif
#ifdef USE_CPU
  bool cpu_mode_available = true;
#else
  bool cpu_mode_available = false;
#endif

}

void print_help(){
std::cerr<<"  Raul P. Pelaez 2019. Fast Mean Square Displacement"<<std::endl;
std::cerr<<"  "<<std::endl;
std::cerr<<"  This code computes the MSD of a list of trajectories in spunto format using the FFT in O(N) time."<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  USAGE:"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  cat pos.dat | msd "<<std::endl;
std::cerr<<"  -N [number of signals (particles)]"<<std::endl;
std::cerr<<"  -Nsteps [number of lags in file]"<<std::endl;
std::cerr<<"  -dimensions [=3, number of columns per signal to read]"<<std::endl;
std::cerr<<"  -device [=auto, choose between CUDA or FFTW: CPU or GPU] "<<std::endl;
std::cerr<<"  -precision [=double, can also be float. Float is much faster and needs half the memory]"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  > pos.msd"<<std::endl;
std::cerr<<"  "<<std::endl;
std::cerr<<"  FORMAT:"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  ---pos.dat---"<<std::endl;
std::cerr<<"  #"<<std::endl;
std::cerr<<"  x1 y1 z1 ... //Coordinates of particle 1 at time 1"<<std::endl;
std::cerr<<"  x2 y2 z2 ..."<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  xN yN zN ..."<<std::endl;
std::cerr<<"  #"<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  ."<<std::endl;
std::cerr<<"  xN yN zN ... //coordinates of particle N at time Nsteps"<<std::endl;
std::cerr<<"  ----"<<std::endl;
std::cerr<<"  EXAMPLE:"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  ---pos.dat---"<<std::endl;
std::cerr<<"  #"<<std::endl;
std::cerr<<"  0 0 0"<<std::endl;
std::cerr<<"  0 0 0"<<std::endl;
std::cerr<<"  #"<<std::endl;
std::cerr<<"  1 1 1"<<std::endl;
std::cerr<<"  1 1 1"<<std::endl;
std::cerr<<"  #"<<std::endl;
std::cerr<<"  2 2 2"<<std::endl;
std::cerr<<"  2 2 2"<<std::endl;
std::cerr<<"  ----"<<std::endl;
std::cerr<<""<<std::endl;
std::cerr<<"  cat pos.dat | msd -N 2 -Nsteps 3 -dimensions 3 > pos.msd"<<std::endl;
    std::cerr<<" Compiled from git commit: "<<GITVERSION<<std::endl;
}

#define fori(x,y) for(int i=x; i<y;i++)

using device = MeanSquareDisplacement::device;  
device dev = device::cpu;
bool force_device = false;
  
std::string fileName;

bool doublePrecision = true;

int signalSize;
int Nsignals;
int dimensions = 3;



void parseCLI(int argc, char *argv[]){
 /* Parse cli input */
  fori(0,argc){
    if(strcmp(argv[i], "-h")==0){
      print_help();
      exit(0);
    }
    else if(strcmp(argv[i], "-N")==0) Nsignals = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-Nsteps")==0) signalSize = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-dimensions")==0) dimensions = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-device")==0){
      if(strcmp(argv[i+1], "GPU") == 0) dev = device::gpu;
      else if(strcmp(argv[i+1], "CPU") == 0) dev = device::cpu;
      else{ std::cerr<<"ERROR!! Unknown device, use CPU or GPU"<<std::endl; print_help(); exit(1);}
      force_device= true;
    }
    else if(strcmp(argv[i], "-precision")==0){
      if(strcmp(argv[i+1], "float") == 0) doublePrecision = false;
      else if(strcmp(argv[i+1], "double") == 0) doublePrecision = true;
      else{ std::cerr<<"ERROR!! Unknown precision, use float or double"<<std::endl; print_help(); exit(1);}
      force_device= true;
    }

  }
  if(!Nsignals || !signalSize){std::cerr<<"ERROR!! SOME INPUT IS MISSING!!!"<<std::endl; print_help(); exit(1);}
  if(dimensions<1){ std::cerr<<"ERROR!! dimensions must be >0"<<std::endl; print_help(); exit(1);}
  
  //Look for the file, in stdin or in a given filename
  if(isatty(STDIN_FILENO)){ //If there is no pipe
    bool good_file = false;
    fori(1,argc){ //exclude the exe file
      std::ifstream in(argv[i]); //There must be a filename somewhere in the cli
      if(in.good()){fileName = std::string(argv[i]);good_file = true;break;}      
    }
    if(!good_file){std::cerr<< "ERROR!, NO INPUT DETECTED!!"<<std::endl; print_help(); exit(1);}
  }
  else fileName = "/dev/stdin"; //If there is something being piped

}

template<class real> void compute_msd(){
  
  //The different coordinates are simply treated as different signals,
  // the only distinction is when writting to disk, were they are printed in different columns
  Nsignals *= dimensions;
    
  std::vector<real> signalOr(signalSize*Nsignals, 0);
  //Read
  {
    superIO::superFile in(fileName);
    if(!in.good()){std::cerr<<"ERROR!: Cannot open file "<<fileName<<std::endl; exit(1);}
    
    std::vector<real> means(Nsignals, 0);
  
    char* line;
    int numberChars;
    
    for(int j = 0; j<signalSize; j++){
      numberChars = in.getNextLine(line);
      for(int i = 0; i<Nsignals; i++){
	double numbers[dimensions];
	if(i%dimensions==0){
	  numberChars = in.getNextLine(line);
	  superIO::string2numbers(line, numberChars, dimensions, numbers);
	}
	signalOr[j+signalSize*i] = numbers[i%dimensions];
	means[i] += signalOr[j+i*signalSize]/signalSize;      
      }
    
    }
    for(int j = 0; j<signalSize; j++){
      fori(0, Nsignals){    
	signalOr[j+i*signalSize]-= means[i];
      }
    }
  }
  //MSD_x(m) = 1/(N-m) \sum_{k=0}^{N-m-1}{(r(k+m) - r(k))^2} =
  //         = 1/(N-m) \sum_{k=0}^{N-m-1}{(r^2(k+m) + r^2(k)} - 2/(N-m)\sum_{k=0}^{N-m-1}{r(k)r(k+m) =
  //         = S1(m) - 2*S2(m)
  //Notice that S2 is the definition of autocorrelation, which can be computed with iFFT(FFT(r)*conj(FFT(r)));
  // if r is padded with zeros to have size 2*N
  
  //Autocorrelation for S2

  if(!force_device){
    if(signalSize*Nsignals > 100000 and
       MeanSquareDisplacement::gpu_mode_available and
       MeanSquareDisplacement::canUseCurrentGPU()) dev = device::gpu;
    else if(MeanSquareDisplacement::cpu_mode_available) dev = device::cpu;
    else{
      std::cerr<<"ERROR! This code was compiled without support for CUDA or FFTW, I need one of them!"<<std::endl;
      exit(1);
    }
    //Something is up with GPU
    dev = device::cpu;
  }
  auto S2s = MeanSquareDisplacement::autocorr<real>(dev, signalOr, signalSize, Nsignals);
  std::vector<real> msd(signalSize*Nsignals, 0);


  //S1 can be written as a self recurrent relation by defining:
  // D(k) = r^2(k), D(-1) = D(N) = 0
  // Q    = 2*\sum_{k=0}^{N-1}{D(k)}
  //And doing for m = 0, ..., N-1:
  // Q = Q - D(m-1) - D(N-m)
  // S1(m) = Q/(N-m)
  
  std::vector<real> D(signalSize*Nsignals, 0); 
  
  std::transform(signalOr.begin(), signalOr.end(), D.begin(), [](real pos){return pos*pos;});

  //This loop computes S1, computes msd = S1- 2*S2 and averages msd for every signal (disambiguating the different coordinates)
  for(int j = 0; j<Nsignals; j++){
    real Q = 2*std::accumulate(D.begin()+j*signalSize, D.begin()+(j+1)*signalSize, 0.0f);
    for(int i = 0; i<signalSize; i++){
      if(i>0)
	Q -= D[j*signalSize+i-1] + D[j*signalSize+signalSize-i];
    
      msd[(j%dimensions)*signalSize+i] +=
	Q/(real(signalSize-i)*(Nsignals/dimensions)) -//S1
	2.0*S2s[j*signalSize+i]/(Nsignals/dimensions); //2*S2
    }    
  }

  //Print results
  std::cout<<0<<" ";
  for(int k = 0; k<dimensions; k++)
    std::cout<<0<<" ";
  std::cout<<"\n";
  
  fori(1, signalSize){
    std::cout<<i<<" ";
    for(int k = 0; k<dimensions; k++)
      std::cout<<msd[i+signalSize*k]<<" ";
    std::cout<<"\n";
  }
  


}

int main(int argc, char *argv[]){

  parseCLI(argc, argv);

  if(doublePrecision)
    compute_msd<double>();
  else
    compute_msd<float>();
  
 return 0;
}
