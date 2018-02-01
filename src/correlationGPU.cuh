/*Raul P. Pelaez 2018. Computes the correlation using cuFFT.
  This file is modified from my FastCorrelation to compute only autocorrelation and being the least memory hungry as posible.

 */
#ifndef CORRELATIONGPU_CUH
#define CORRELATIONGPU_CUH


#include"common.h"
#include<cufft.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include"superRead.h"
#include"cufftPrecisionAgnostic.h"

namespace FastCorrelation{

  template<class real>
  __global__ void convolution(cufftComplex_t<real> *A,
			      cufftComplex_t<real> *output, int N, real prefactor){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;

    const auto a = A[i];
    output[i].x =(a.x*a.x+a.y*a.y)*prefactor;
    output[i].y = real(0.0);
  }

  template<class real>
  void autocorrelationGPUFFT(cufftReal_t<real> *A,
			 int numberElements,
			 int nsignals,
			 int maxLag,
			 ScaleMode scaleMode){

    using cufftComplex_t = cufftComplex_t<real>;
    using cufftReal_t = cufftReal_t<real>;

    //Each signal is duplicated and the second part filled with zeros, this way the results are equivalent to the usual O(N^2) correlation algorithm.
    int numberElementsPadded = numberElements + maxLag;
    
    thrust::device_vector<cufftComplex_t> d_signalA(numberElementsPadded*nsignals, cufftComplex_t());
    
    cufftComplex_t *d_signalA_ptr = thrust::raw_pointer_cast(d_signalA.data());

    cudaMemcpy(d_signalA_ptr, A, numberElements*nsignals*sizeof(cufftReal_t), cudaMemcpyHostToDevice);

    
    cudaStream_t stream;
    cudaStreamCreate(&stream);


    //FFT both signals
    cufftHandle plan;
    {
      //Special plan for interleaved signals
      int rank = 1;           // --- 1D FFTs
      int n[] = { numberElementsPadded };   // --- Size of the Fourier transform
      int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
      int idist = 1, odist = 1; // --- Distance between batches
      int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
      int batch = nsignals;     // --- Number of batched executions
      cufftPlanMany(&plan, rank, n, 
		    inembed, istride, idist,
		    onembed, ostride, odist, CUFFT_Real2Complex<real>::value, batch);
    }
    cufftSetStream(plan, stream);
    cufftExecReal2Complex<real>(plan, (cufftReal_t*) d_signalA_ptr, d_signalA_ptr);


    //Convolve TODO: there are more blocks than necessary
    int Nthreads=512;
    int Nblocks =(nsignals*(numberElementsPadded+1))/Nthreads+1;
    convolution<real><<<Nblocks, Nthreads, 0, stream>>>(d_signalA_ptr,
							d_signalA_ptr,
							nsignals*(numberElementsPadded+1),
							1/(double(numberElements)));

    //Inverse FFT the convolution to obtain correlation
    cufftHandle plan2;
    {
      //Special plan for interleaved signals
      int rank = 1;                           // --- 1D FFTs
      int n[] = { numberElementsPadded };                 // --- Size of the Fourier transform
      int istride = nsignals, ostride = nsignals;        // --- Distance between two successive input/output elements
      int idist = 1, odist = 1; // --- Distance between batches
      int inembed[] = { 0 };          // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = { 0 };         // --- Output size with pitch (ignored for 1D transforms)
      int batch = nsignals;                      // --- Number of batched executions
      cufftPlanMany(&plan2, rank, n, 
		    inembed, istride, idist,
		    onembed, ostride, odist, CUFFT_Complex2Real<real>::value, batch);
    }

    cufftSetStream(plan2, stream);
    cufftExecComplex2Real<real>(plan2, d_signalA_ptr, (cufftReal_t*)d_signalA_ptr);
    //Download
    //h_signalA = d_signalA; //signalA now holds correlation
    cudaMemcpy(A, d_signalA_ptr, numberElements*nsignals*sizeof(cufftReal_t), cudaMemcpyDeviceToHost);
    
    cufftReal_t* h_signalA_ptr = (cufftReal_t*) A;//thrust::raw_pointer_cast(h_signalA.data());

    //Write results
    for(int i = 0; i<maxLag; i++){
      double mean = 0.0;
      //double mean2 = 0.0;
      
      double scale;
      switch(scaleMode){
      case ScaleMode::biased:   scale = 1; break;
      case ScaleMode::unbiased: scale = double(maxLag)/(maxLag-i+1); break;
      case ScaleMode::none:     scale = maxLag; break;
      default: scale = 1; break;
      }
      //I really do not know where this comes from, but it works... I think I messed some normalization in the FFT
      double misteriousScale =0.5*(2-double(maxLag)/numberElements);
      
      
      for(int s=0; s<nsignals;s++){ //Average all correlations
	double tmp = h_signalA_ptr[nsignals*i+s];
	
	mean = tmp/numberElements;

	double corr = mean*misteriousScale*scale;

	((cufftReal_t*)A)[i*nsignals+s] = corr;

      }
      // mean  /= double(nsignals)*numberElements;
      // mean2 /= double(nsignals)*numberElements*numberElements;
      
      // double corr = mean*misteriousScale*scale;
      // double error = sqrt(mean2 - mean*mean)*misteriousScale*scale;
      // corrout[i] = corr;

    }

    cufftDestroy(plan);
    cufftDestroy(plan2);
    cudaStreamDestroy(stream);
  }
}
#endif