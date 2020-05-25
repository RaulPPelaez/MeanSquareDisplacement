/*Raul P. Pelaez 2019. Fast Mean Square Displacement

  This code computes the MSD of a list of trajectories in spunto format using
the FFT in O(N) time.


  USAGE:

  cat pos.dat | msd
  -N [number of signals (particles)]
  -Nsteps [number of lags in file]
  -dimensions [=3, number of columns per signal to read]
  -device [=auto, choose between CUDA or FFTW: CPU or GPU]
  -precision [=double, can also be float. Float is much faster and needs half
the memory]

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


  THEORY:
  MSD_x(m) = 1/(N-m) \sum_{k=0}^{N-m-1}{(r(k+m) - r(k))^2} =
           = 1/(N-m) \sum_{k=0}^{N-m-1}{(r^2(k+m) + r^2(k)} - 2/(N-m)\sum_{k=0}^{N-m-1}{r(k)r(k+m) =
           = S1(m) - 2*S2(m)
  Notice that S2 is the definition of autocorrelation, which can be computed with iFFT(FFT(r)*conj(FFT(r)));
  if r is padded with zeros to have size 2*N
  S1 can be written as a self recurrent relation by defining:
  D(k) = r^2(k), D(-1) = D(N) = 0
  Q    = 2*\sum_{k=0}^{N-1}{D(k)}
  And doing for m = 0, ..., N-1:
  Q = Q - D(m-1) - D(N-m)
  S1(m) = Q/(N-m)  
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
  namespace detail{

    template<class real>
    struct KahanAccumulation{
      real sum = 0;
      real correction = 0;
    };
  
    template<class real>
    KahanAccumulation<real> KahanSum(KahanAccumulation<real> accumulation, real value){
      KahanAccumulation<real> result;
      real y = value - accumulation.correction;
      real t = accumulation.sum + y;
      result.correction = (t - accumulation.sum) - y;
      result.sum = t;
      return result;
    }
  }  
  template<class real, class Iterator>
  real safeAccumulate(Iterator begin, Iterator end, real init = real()){
    detail::KahanAccumulation<real> initV;
    initV.sum = init;
    auto res = std::accumulate(begin, end, initV, detail::KahanSum<real>);    
    return res.sum;
  }
  
}

#define fori(x,y) for(int i=x; i<y;i++)

using device = MeanSquareDisplacement::device;

struct Configuration{
  device dev = device::cpu;
  bool force_device = false;
  std::string fileName;
  bool doublePrecision = true;
  int signalSize = -1;
  int Nsignals = -1;
  int dimensions = 3;
};

Configuration parseCLI(int argc, char *argv[]);

void print_help();

template <class real>
std::vector<real> readSignal(std::string fileName, int signalSize, int Nsignals, int dimensions) {
  std::vector<real> signal(signalSize*Nsignals*dimensions);
  superIO::superInputFile in(fileName);
  if(!in.good()){
    std::cerr<<"ERROR!: Cannot read from "<<fileName<<std::endl;
    exit(1);
  }    
  char* line;
  int numberChars;
  for(int j = 0; j<signalSize; j++){
    numberChars = in.getNextLine(line);
    for(int i = 0; i<Nsignals*dimensions; i++){
      double numbers[dimensions];
      if(i%dimensions==0){
	numberChars = in.getNextLine(line);
	superIO::string2numbers(line, numberChars, dimensions, numbers);
      }
      signal[j+signalSize*i] = numbers[i%dimensions];
    }
  }
  return std::move(signal);
}
  
template <class real>
void substractMean(std::vector<real> &signal, int Nsignals, int signalSize) {
  std::vector<real> mean(Nsignals*signalSize, 0);
  for(int i=0; i<Nsignals; i++){
    auto first = signal.begin() + signalSize*i;
    real mean = MeanSquareDisplacement::safeAccumulate<real>(first, first + signalSize)/signalSize;
    std::transform(first, first + signalSize, first, [&](real s){ return s - mean; });
  }
}

device chooseDevice(Configuration conf){
  device dev = conf.dev;
  if(!conf.force_device){
    if((conf.signalSize*conf.Nsignals*conf.dimensions > 1000000 or not MeanSquareDisplacement::cpu_mode_available) and
       MeanSquareDisplacement::gpu_mode_available and
       MeanSquareDisplacement::canUseCurrentGPU()){
      dev = device::gpu;
    }
    else if(MeanSquareDisplacement::cpu_mode_available){
      dev = device::cpu;
    }
    else{
      std::cerr<<"ERROR! This code was compiled without support for CUDA or FFTW, I need one of them!"<<std::endl;
      exit(1);
    }
  }
  return dev;
}

template <class real>
std::vector<real> computeS1(std::vector<real> &signal, int signalSize, int Nsignals, int dimensions) {
  std::vector<real> D(signalSize*dimensions);
  std::vector<real> S1(signalSize*dimensions);
  for(int j = 0; j<Nsignals*dimensions; j++){
    auto signali = signal.begin() + signalSize*j;
    std::transform(signali, signali + signalSize, D.begin(), [](real s){return s*s;});
    real Q = 2.0*MeanSquareDisplacement::safeAccumulate<real>(D.begin(), D.end());
    for(int i = 0; i<signalSize; i++){
      if(i>0){
	Q -= D[i-1] + D[signalSize-i];
      }
      S1[i+signalSize*(j%dimensions)] += Q/(real(signalSize-i)*Nsignals);
    }
  }
  return std::move(S1);
}

template <class real>
std::vector<real> computeS2(std::vector<real> &signal, Configuration conf){  
  auto dev = chooseDevice(conf);
  //The different coordinates are simply treated as different signals  
  auto S2 = MeanSquareDisplacement::autocorr<real>(dev, signal, conf.signalSize, conf.Nsignals*conf.dimensions);
  std::vector<real> s2Averaged(conf.signalSize*conf.dimensions);
  std::vector<real> sumAux(conf.Nsignals*conf.dimensions);
  for(int i = 0; i<conf.signalSize; i++){
    for(int j = 0; j< conf.Nsignals*conf.dimensions; j++){
      sumAux[j/conf.dimensions+conf.Nsignals*(j%conf.dimensions)] = S2[i + j*conf.signalSize]/conf.Nsignals;
    }
    for(int j = 0; j< conf.dimensions; j++){
      auto first = sumAux.begin() + j*conf.Nsignals;
      auto sum = MeanSquareDisplacement::safeAccumulate<real>(first, first + conf.Nsignals);      
      s2Averaged[i+conf.signalSize*j] = sum;
    }    
  }
  return std::move(s2Averaged);
}

template<class real> std::vector<real> computeMSD(std::vector<real> &signal, Configuration conf){
  substractMean(signal, conf.Nsignals*conf.dimensions, conf.signalSize);
  auto S2 = computeS2(signal, conf);
  auto S1 = computeS1(signal, conf.signalSize, conf.Nsignals, conf.dimensions);
  std::vector<real> msd(conf.signalSize*conf.dimensions);
  std::transform(S1.begin(), S1.end(), S2.begin(), msd.begin(),
   		 [](real S1, real S2){
   		   return (S1 - 2.0*S2);
   		 }
   		 );
  return std::move(msd);
}

template <class real>
void writeMSD(std::vector<real> &msd, Configuration conf) {
  std::cout.precision(sizeof(real)*2);
  std::cout<<0<<" ";
  for(int k = 0; k<conf.dimensions; k++){
    std::cout<<0<<" ";
  }
  std::cout<<"\n";  
  fori(1, conf.signalSize){
    std::cout<<i<<" ";
    for(int k = 0; k<conf.dimensions; k++)
      std::cout<<msd[i+conf.signalSize*k]<<" ";
    std::cout<<"\n";
  }    
}

template<class real> void run(Configuration conf){
  auto signal = readSignal<real>(conf.fileName, conf.signalSize, conf.Nsignals, conf.dimensions);
  auto msd = computeMSD(signal, conf);
  writeMSD(msd, conf);
}

int main(int argc, char *argv[]){
  auto conf = parseCLI(argc, argv);
  if(conf.doublePrecision){
    run<double>(conf);
  }
  else{
    run<float>(conf);
  } 
 return 0;
}

Configuration parseCLI(int argc, char *argv[]){
  Configuration conf;
  fori(0,argc){
    if(strcmp(argv[i], "-h")==0){
      print_help();
      exit(0);
    }
    else if(strcmp(argv[i], "-N")==0) conf.Nsignals = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-Nsteps")==0) conf.signalSize = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-dimensions")==0) conf.dimensions = atoi(argv[i+1]);
    else if(strcmp(argv[i], "-device")==0){
      if(strcmp(argv[i+1], "GPU") == 0) conf.dev = device::gpu;
      else if(strcmp(argv[i+1], "CPU") == 0) conf.dev = device::cpu;
      else{
	std::cerr<<"ERROR!! Unknown device, use CPU or GPU"<<std::endl;
	print_help();
	exit(1);
      }
      conf.force_device= true;
    }
    else if(strcmp(argv[i], "-precision")==0){
      if(strcmp(argv[i+1], "float") == 0) conf.doublePrecision = false;
      else if(strcmp(argv[i+1], "double") == 0) conf.doublePrecision = true;
      else{
	std::cerr<<"ERROR!! Unknown precision, use float or double"<<std::endl;
	print_help();
	exit(1);
      }
      conf.force_device= true;
    }
  }
  if(!conf.Nsignals || !conf.signalSize){
    std::cerr<<"ERROR!! SOME INPUT IS MISSING!!!"<<std::endl;
    print_help();
    exit(1);
  }
  if(conf.dimensions<1){ std::cerr<<"ERROR!! dimensions must be >0"<<std::endl;
    print_help();
    exit(1);
  }
  
  //Look for the file, in stdin or in a given filename
  if(isatty(STDIN_FILENO)){ //If there is no pipe
    bool good_file = false;
    fori(1,argc){
      std::ifstream in(argv[i]); //There must be a filename somewhere in the cli
      if(in.good()){
	conf.fileName = std::string(argv[i]);
	good_file = true;
	break;
      }      
    }
    if(!good_file){
      std::cerr<< "ERROR!, NO INPUT DETECTED!!"<<std::endl;
      print_help();
      exit(1);
    }
  }
  else conf.fileName = "/dev/stdin"; //If there is something being piped
  return conf;
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
