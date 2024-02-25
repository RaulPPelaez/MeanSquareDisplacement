#ifndef AUTOCORRGPU_CUH
#define AUTOCORRGPU_CUH

#ifdef USE_GPU
#include<vector>
#include<thrust/device_vector.h>
#include"cufftPrecisionAgnostic.h"
#include"cufftDebug.h"
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <thrust/device_malloc_allocator.h>

namespace MeanSquareDisplacement{


  bool canUseCurrentGPU(){
    constexpr int minArch = 35;

    int dev = -1;
    CudaSafeCall(cudaGetDevice(&dev));

    cudaDeviceProp deviceProp;
    CudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));

    int cuda_arch = 100*deviceProp.major + 10*deviceProp.minor;
    if(cuda_arch < minArch) return false;
    return true;
  }
  template<class T>
  class managed_allocator : public thrust::device_malloc_allocator<T>{
  public:
    using value_type = T;
    typedef thrust::device_ptr<T>  pointer;
    inline pointer allocate(size_t n){
      value_type* result = nullptr;
      cudaError_t error = cudaMallocManaged(&result, sizeof(T)*n, cudaMemAttachGlobal);
      if(error != cudaSuccess)
	throw thrust::system_error(error, thrust::cuda_category(),
				   "managed_allocator::allocate(): cudaMallocManaged");
      return thrust::device_pointer_cast(result);
    }

    inline void deallocate(pointer ptr, size_t){
      cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));
      if(error != cudaSuccess)
	throw thrust::system_error(error, thrust::cuda_category(),
				   "managed_allocator::deallocate(): cudaFree");
    }
  };

  template<class T>
  using managed_vector = thrust::device_vector<T, managed_allocator<T>>;


  template<class real>
  std::vector<real> autocorrGPU(std::vector<real> &signalOr, int signalSize, int Nsignals){

    using cufftReal = cufftReal_t<real>;
    using cufftComplex = cufftComplex_t<real>;
    //pad with zeros
    int signalSizePad =signalSize*2;

    int N = 2*(signalSizePad/2+1);
    //thrust::host_vector<real> signalH;
    managed_vector<real> signal;
    try{
      signal.resize(Nsignals*N+2, 0);
    }
    catch(thrust::system_error &e){
      std::cerr<<"Thrust host vector creation failed with error: "<<e.what()<<std::endl;
      exit(1);
    }
    {
      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      for(int i = 0; i<Nsignals; i++){
	for(int j = 0; j<signalSize; j++){
	  signal_ptr[j+i*N] = signalOr[j+i*(signalSize)];
	}
      }
    }

    /*
    thrust::device_vector<real> signal;
    try{
      signal = signalH;
    }
    catch(thrust::system_error &e){
      std::cerr<<"Thrust device vector creation failed with error: "<<e.what()<<std::endl;
      exit(1);
    }
    */
    size_t cufft_storage = 0;

    managed_vector<char> tmp_storage;
    {
      //FFT both signals
      cufftHandle plan;
      {
	CufftSafeCall(cufftCreate(&plan));
	CufftSafeCall(cufftSetAutoAllocation(plan, 0));
	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = N, odist = signalSizePad/2+1; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
	CufftSafeCall(cufftMakePlanMany(plan, rank, n,
					inembed, istride, idist,
					onembed, ostride, odist, CUFFT_Real2Complex<real>::value, batch,
					&cufft_storage));

	// size_t free_mem, total_mem;
	// CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));

	// if(free_mem < cufft_storage){
	//   std::cerr<<"Not enough memory in device!"<<std::endl;
	//   exit(1);
	// }
	try{
	  tmp_storage.resize(cufft_storage);
	}
	catch(thrust::system_error &e){
	  std::cerr<<"Thrust could not resize vector with error: "<<e.what()<<std::endl;
	  exit(1);
	}
	CufftSafeCall(cufftSetWorkArea(plan,
				       (void*)thrust::raw_pointer_cast(tmp_storage.data())));

      }


      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      CufftSafeCall(cufftExecReal2Complex<real>(plan, (cufftReal*)signal_ptr, (cufftComplex*)signal_ptr));
      CufftSafeCall(cufftDestroy(plan));
    }
    {
      try{
	thrust::device_ptr<cufftComplex> signal_complex((cufftComplex*)thrust::raw_pointer_cast(signal.data()));
	thrust::transform(thrust::cuda::par,
			  signal_complex, signal_complex+signal.size()/2, signal_complex,
			  [=] __device__ (cufftComplex a){
			    return cufftComplex{(a.x*a.x+a.y*a.y)/(real)signalSizePad, 0};
			  }
			  );
      }
      catch(thrust::system_error &e){
	std::cerr<<"Thrust transform failed with error: "<<e.what()<<std::endl;
	exit(1);
      }

    }
    {
      //FFT both signals
      cufftHandle plan;
      {
	cufft_storage = 0;
	CufftSafeCall(cufftCreate(&plan));
	CufftSafeCall(cufftSetAutoAllocation(plan, 0));

	//Special plan for interleaved signals
	int rank = 1;           // --- 1D FFTs
	int n[] = { signalSizePad };   // --- Size of the Fourier transform
	int istride = 1, ostride = 1;        // --- Distance between two successive input/output elements
	int idist = signalSizePad/2+1, odist = N; // --- Distance between batches
	int inembed[] = { 0 };    // --- Input size with pitch (ignored for 1D transforms)
	int onembed[] = { 0 };    // --- Output size with pitch (ignored for 1D transforms)
	int batch = Nsignals;     // --- Number of batched executions
	CufftSafeCall(cufftMakePlanMany(plan, rank, n,
					inembed, istride, idist,
					onembed, ostride, odist, CUFFT_Complex2Real<real>::value, batch,
					&cufft_storage));
	tmp_storage.resize(cufft_storage);
	CufftSafeCall(cufftSetWorkArea(plan,
				       (void*)thrust::raw_pointer_cast(tmp_storage.data())));

      }
      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      CufftSafeCall(cufftExecComplex2Real<real>(plan, (cufftComplex*)signal_ptr, (cufftReal*)signal_ptr));
      CufftSafeCall(cufftDestroy(plan));
      tmp_storage.clear();
    }
    std::vector<real> res(signalSize*Nsignals);
    // try{
    //   signalH = signal;
    // }
    // catch(thrust::system_error &e){
    //   std::cerr<<"Thrust D2H copy failed with error: "<<e.what()<<std::endl;
    //   exit(1);
    // }
    CudaSafeCall(cudaDeviceSynchronize());
    {
      auto signal_ptr = thrust::raw_pointer_cast(signal.data());
      for(int i = 0; i<Nsignals; i++){
	for(int j = 0; j<signalSize; j++){
	  res[i*signalSize+j] = signal_ptr[i*N+j]/(signalSize-j);
	}
      }
    }
    return res;
  }
}
#else
#include<vector>
#include<iostream>

namespace MeanSquareDisplacement{
  bool canUseCurrentGPU(){
    return false;
  }
  template<class real, class ...T>
  std::vector<real> autocorrGPU(T...){
    std::cerr<<"THIS CODE WAS COMPILED IN CPU ONLY MODE"<<std::endl;
    exit(1);
  }
}
#endif

#endif
