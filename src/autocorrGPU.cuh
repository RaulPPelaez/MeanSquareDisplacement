

#ifdef USE_GPU
#include<vector>
#include<thrust/device_vector.h>
#include"cufftPrecisionAgnostic.h"
namespace MeanSquareDisplacement{
  template<class real>
  std::vector<real> autocorrGPU(std::vector<real> &signalOr, int signalSize, int Nsignals){
    
    using cufftReal = cufftReal_t<real>;
    using cufftComplex = cufftComplex_t<real>;
    //pad with zeros
    int signalSizePad =signalSize*2;
  
    int N = 2*(signalSizePad/2+1);
  
    thrust::host_vector<real> signalH(Nsignals*N+2, 0);
  
    for(int i = 0; i<Nsignals; i++){
      for(int j = 0; j<signalSize; j++){
	signalH[j+i*N] = signalOr[j+i*(signalSize)];
      }
    }
  
    thrust::device_vector<real> signal = signalH;
  
    {
      //FFT both signals
      cufftHandle plan;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = N, odist = signalSizePad/2+1; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
	cufftPlanMany(&plan, rank, n, 
		      inembed, istride, idist,
		      onembed, ostride, odist, CUFFT_Real2Complex<real>::value, batch);
      }
  

      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      cufftExecReal2Complex<real>(plan, (cufftReal*)signal_ptr, (cufftComplex*)signal_ptr);
      cufftDestroy(plan);
    }
    {
      thrust::device_ptr<cufftComplex> signal_complex((cufftComplex*)thrust::raw_pointer_cast(signal.data()));

      thrust::transform(signal_complex, signal_complex+signal.size()/2, signal_complex,
			[=] __device__ (cufftComplex a){
			  return cufftComplex{(a.x*a.x+a.y*a.y)/(real)signalSizePad, 0};
			}
			);
    }
    {
      //FFT both signals
      cufftHandle plan;
      {
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = signalSizePad/2+1, odist = N; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
	cufftPlanMany(&plan, rank, n, 
		      inembed, istride, idist,
		      onembed, ostride, odist, CUFFT_Complex2Real<real>::value, batch);
      }
      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      cudaDeviceSynchronize();
      cufftExecComplex2Real<real>(plan, (cufftComplex*)signal_ptr, (cufftReal*)signal_ptr);
      cufftDestroy(plan);
    }
		    
    std::vector<real> res(signalSize*Nsignals);
    signalH = signal;
    for(int i = 0; i<Nsignals; i++){  
      for(int j = 0; j<signalSize; j++){
	res[i*signalSize+j] = signalH[i*N+j]/(signalSize-j);
      }
    }
  

    return res;
  }
}
#else
#include<vector>
#include<iostream>

namespace MeanSquareDisplacement{
  template<class real, class ...T>
  std::vector<real> autocorrGPU(T...){
    std::cerr<<"THIS CODE WAS COMPILED IN CPU ONLY MODE"<<std::endl;
    exit(1);     
  }
}
#endif
