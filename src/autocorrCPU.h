
#ifdef USE_CPU
#include"fftwPrecisionAgnostic.h"
#include<vector>
namespace MeanSquareDisplacement{
  template<class real>
  std::vector<real> autocorrCPU(std::vector<real> &signalOr, int signalSize, int Nsignals){
    using fftw_complex = fftw_complex_t<real>;
    using fftw_plan = fftw_plan_t<real>;
  
    //pad with zeros
    int signalSizePad =signalSize*2;
  
    int N = 2*(signalSizePad/2+1);
	
    std::vector<real, FFTWallocator<real>> signal(Nsignals*N+2, 0);
  
    for(int i = 0; i<Nsignals; i++){
      for(int j = 0; j<signalSize; j++){
	signal[j+i*N] = signalOr[j+i*(signalSize)];
      }
    }
  
  
    {
      fftw_plan plan;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = N, odist = signalSizePad/2+1; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
      
	plan = fftw_plan_many_dft_r2c_prec<real>()(rank, n, batch,
						   (real*)signal.data(), inembed, istride, idist,
						   (fftw_complex*)signal.data(), onembed, ostride, odist,
						   FFTW_ESTIMATE);

      }
      MeanSquareDisplacement::fftw_execute(plan);
    }
    {
      fftw_complex* signal_complex = (fftw_complex*)(signal.data());

      for(int i=0;i<signal.size()/2; i++){
	fftw_complex a;
	a[0] = signal_complex[i][0];
	a[1] = signal_complex[i][1];
	fftw_complex b;
	b[1] = 0;
	b[0] =(a[0]*a[0]+a[1]*a[1])/(real)signalSizePad;
	signal_complex[i][0] = b[0];
	signal_complex[i][1] = b[1];
      
  
      }
    }
    {
      fftw_plan plan;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = signalSizePad/2+1, odist = N; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
	plan = fftw_plan_many_dft_c2r_prec<real>()(rank, n, batch,
						   (fftw_complex*)signal.data(), inembed, istride, idist,
						   (real*)signal.data(), onembed, ostride, odist,
						   FFTW_ESTIMATE);

      }
      auto signal_ptr = signal.data();
      MeanSquareDisplacement::fftw_execute(plan);
    }
		    
    std::vector<real> res(signalSize*Nsignals);  
    for(int i = 0; i<Nsignals; i++){  
      for(int j = 0; j<signalSize; j++){
	res[i*signalSize+j] = signal[i*N+j]/(signalSize-j);
      }
    }
  
    return res;
  }


}



#else
#include<vector>
#include<iostream>
namespace MeanSquareDisplacement{
  template<class real>
  std::vector<real> autocorrCPU(std::vector<real> &signalOr, int signalSize, int Nsignals){
    std::cerr<<"THIS CODE WAS NOT COMPILED IN CPU MODE"<<std::endl;
    exit(1);     
  }

}

#endif


