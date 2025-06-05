/*Raul P. Pelaez. Fast Mean Square Displacement

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
           = 1/(N-m) \sum_{k=0}^{N-m-1}{(r^2(k+m) + r^2(k)} -
2/(N-m)\sum_{k=0}^{N-m-1}{r(k)r(k+m) = = S1(m) - 2*S2(m) Notice that S2 is the
definition of autocorrelation, which can be computed with
iFFT(FFT(r)*conj(FFT(r))); if r is padded with zeros to have size 2*N S1 can be
written as a self recurrent relation by defining: D(k) = r^2(k), D(-1) = D(N) =
0 Q    = 2*\sum_{k=0}^{N-1}{D(k)} And doing for m = 0, ..., N-1: Q = Q - D(m-1)
- D(N-m) S1(m) = Q/(N-m)
 */
#include "msd/defines.h"
#include "msd/gitversion.h"
#include "msd/msd.hpp"
#include "msd/superIO.h"
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>
using namespace msd;

struct Configuration {
  std::string fileName;
  bool doublePrecision = true;
  int signalSize = -1;
  int Nsignals = -1;
  int dimensions = 3;
  device dev = device::none;
};

Configuration parseCLI(int argc, char *argv[]);

void print_help();

template <class real>
std::vector<real> readSignal(std::string fileName, int signalSize, int Nsignals,
                             int dimensions) {
  std::vector<real> signal(signalSize * Nsignals * dimensions);
  superIO::superInputFile in(fileName);
  if (!in.good()) {
    std::cerr << "ERROR!: Cannot read from " << fileName << std::endl;
    exit(1);
  }
  char *line;
  int numberChars;
  // The file looks like:
  // # (comment)
  // x1 y1 z1 ... //Coordinates of particle 1 at time 1
  // x2 y2 z2 ... //Coordinates of particle 2 at time 1
  // ...
  // #
  // x1 y1 z1 ... //Coordinates of particle 1 at time 2
  // x2 y2 z2 ... //Coordinates of particle 2 at time 2
  // ...
  for (int j = 0; j < signalSize; j++) {
    numberChars = in.getNextLine(line);
    for (int i = 0; i < Nsignals * dimensions; i++) {
      double numbers[dimensions];
      if (i % dimensions == 0) {
        numberChars = in.getNextLine(line);
        superIO::string2numbers(line, numberChars, dimensions, numbers);
      }
      signal[j + signalSize * i] = numbers[i % dimensions];
    }
  }
  return std::move(signal);
}

template <class real>
void writeMSD(std::vector<real> &msd, Configuration conf) {
  std::cout.precision(sizeof(real) * 2);
  std::cout << 0 << " ";
  for (int k = 0; k < conf.dimensions; k++) {
    std::cout << 0 << " ";
  }
  std::cout << "\n";
  for (int i = 1; i < conf.signalSize; i++) {
    std::cout << i << " ";
    for (int k = 0; k < conf.dimensions; k++)
      std::cout << msd[i + conf.signalSize * k] << " ";
    std::cout << "\n";
  }
}

template <class real> void run(Configuration conf) {
  auto signal = readSignal<real>(conf.fileName, conf.signalSize, conf.Nsignals,
                                 conf.dimensions);
  std::span<real> signal_span(signal);

  auto msd = mean_square_displacement(signal_span, conf.dev, conf.Nsignals,
                                      conf.signalSize, conf.dimensions);
  writeMSD(msd, conf);
}

int main(int argc, char *argv[]) {
  auto conf = parseCLI(argc, argv);
  if (conf.doublePrecision) {
    run<double>(conf);
  } else {
    run<float>(conf);
  }
  return 0;
}

Configuration parseCLI(int argc, char *argv[]) {
  Configuration conf;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      print_help();
      exit(0);
    } else if (strcmp(argv[i], "-N") == 0)
      conf.Nsignals = atoi(argv[i + 1]);
    else if (strcmp(argv[i], "-Nsteps") == 0)
      conf.signalSize = atoi(argv[i + 1]);
    else if (strcmp(argv[i], "-dimensions") == 0)
      conf.dimensions = atoi(argv[i + 1]);
    else if (strcmp(argv[i], "-device") == 0) {
      if (strcmp(argv[i + 1], "cpu") == 0)
        conf.dev = device::cpu;
      else if (strcmp(argv[i + 1], "gpu") == 0)
        conf.dev = device::gpu;
      else if (strcmp(argv[i + 1], "auto") == 0)
        conf.dev = device::none; // Let the code choose the best device
      else {
        std::cerr << "ERROR!! Unknown device, use cpu or gpu" << std::endl;
        print_help();
        exit(1);
      }
    } else if (strcmp(argv[i], "-precision") == 0) {
      if (strcmp(argv[i + 1], "float") == 0)
        conf.doublePrecision = false;
      else if (strcmp(argv[i + 1], "double") == 0)
        conf.doublePrecision = true;
      else {
        std::cerr << "ERROR!! Unknown precision, use float or double"
                  << std::endl;
        print_help();
        exit(1);
      }
    }
  }
  if (!conf.Nsignals || !conf.signalSize) {
    std::cerr << "ERROR!! SOME INPUT IS MISSING!!!" << std::endl;
    print_help();
    exit(1);
  }
  if (conf.dimensions < 1) {
    std::cerr << "ERROR!! dimensions must be >0" << std::endl;
    print_help();
    exit(1);
  }

  // Look for the file, in stdin or in a given filename
  if (isatty(STDIN_FILENO)) { // If there is no pipe
    bool good_file = false;
    for (int i = 1; i < argc; i++) {
      std::ifstream in(argv[i]); // There must be a filename somewhere in the
                                 // cli
      if (in.good()) {
        conf.fileName = std::string(argv[i]);
        good_file = true;
        break;
      }
    }
    if (!good_file) {
      std::cerr << "ERROR!, NO INPUT DETECTED!!" << std::endl;
      print_help();
      exit(1);
    }
  } else
    conf.fileName = "/dev/stdin"; // If there is something being piped
  return conf;
}

void print_help() {
  std::cerr << "  Raul P. Pelaez. Fast Mean Square Displacement" << std::endl;
  std::cerr << " Project version: " << MSD_VERSION << std::endl;
  std::cerr << "  " << std::endl;
  std::cerr << "  This code computes the MSD of a list of trajectories in "
               "spunto format using the FFT in O(N) time."
            << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  USAGE:" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  cat pos.dat | msd " << std::endl;
  std::cerr << "  -N [number of signals (particles)]" << std::endl;
  std::cerr << "  -Nsteps [number of lags in file]" << std::endl;
  std::cerr << "  -dimensions [=3, number of columns per signal to read]"
            << std::endl;
  std::cerr << "  -precision [=double, can also be float. Float is much faster "
               "and needs half the memory]"
            << std::endl;
  std::cerr << "  -device [=auto, choose between gpu or cpu" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  > pos.msd" << std::endl;
  std::cerr << "  " << std::endl;
  std::cerr << "  FORMAT:" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  ---pos.dat---" << std::endl;
  std::cerr << "  #" << std::endl;
  std::cerr << "  x1 y1 z1 ... //Coordinates of particle 1 at time 1"
            << std::endl;
  std::cerr << "  x2 y2 z2 ..." << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  xN yN zN ..." << std::endl;
  std::cerr << "  #" << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  ." << std::endl;
  std::cerr << "  xN yN zN ... //coordinates of particle N at time Nsteps"
            << std::endl;
  std::cerr << "  ----" << std::endl;
  std::cerr << "  EXAMPLE:" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  ---pos.dat---" << std::endl;
  std::cerr << "  #" << std::endl;
  std::cerr << "  0 0 0" << std::endl;
  std::cerr << "  0 0 0" << std::endl;
  std::cerr << "  #" << std::endl;
  std::cerr << "  1 1 1" << std::endl;
  std::cerr << "  1 1 1" << std::endl;
  std::cerr << "  #" << std::endl;
  std::cerr << "  2 2 2" << std::endl;
  std::cerr << "  2 2 2" << std::endl;
  std::cerr << "  ----" << std::endl;
  std::cerr << "" << std::endl;
  std::cerr << "  cat pos.dat | msd -N 2 -Nsteps 3 -dimensions 3 > pos.msd"
            << std::endl;
  std::cerr << " Compiled from git commit: " << GITVERSION << std::endl;
}
