#include "msd/autocorr_cuda.cuh"
#include "msd/cufftDebug.h"
#include "msd/cufftPrecisionAgnostic.h"
#include <span>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <vector>
namespace msd {

template <class T>
class managed_allocator : public thrust::device_malloc_allocator<T> {
public:
  using value_type = T;
  typedef thrust::device_ptr<T> pointer;
  inline pointer allocate(size_t n) {
    value_type *result = nullptr;
    cudaError_t error =
        cudaMallocManaged(&result, sizeof(T) * n, cudaMemAttachGlobal);
    if (error != cudaSuccess)
      throw thrust::system_error(
          error, thrust::cuda_category(),
          "managed_allocator::allocate(): cudaMallocManaged");
    return thrust::device_pointer_cast(result);
  }

  inline void deallocate(pointer ptr, size_t) {
    cudaError_t error = cudaFree(thrust::raw_pointer_cast(ptr));
    if (error != cudaSuccess)
      throw thrust::system_error(error, thrust::cuda_category(),
                                 "managed_allocator::deallocate(): cudaFree");
  }
};

template <class T>
using managed_vector = thrust::device_vector<T, managed_allocator<T>>;

template <class real>
std::vector<real> autocorr_cuda(const std::span<real> &isignal, int signal_size,
                                int number_signals) {
  using cufftReal = cufftReal_t<real>;
  using cufftComplex = cufftComplex_t<real>;
  // pad with zeros
  int padded_signal_size = signal_size * 2;
  int regularized_size = 2 * (padded_signal_size / 2 + 1);
  managed_vector<real> signal;
  signal.resize(number_signals * regularized_size + 2, 0);
  {
    auto signal_ptr = thrust::raw_pointer_cast(signal.data());
    for (int i = 0; i < number_signals; i++) {
      for (int j = 0; j < signal_size; j++) {
        signal_ptr[j + i * regularized_size] = isignal[j + i * (signal_size)];
      }
    }
  }
  size_t cufft_storage = 0;

  managed_vector<char> tmp_storage;
  {
    // FFT both signals
    cufftHandle plan;
    {
      CufftSafeCall(cufftCreate(&plan));
      CufftSafeCall(cufftSetAutoAllocation(plan, 0));
      // Special plan for interleaved signals
      int rank = 1;                   // --- 1D FFTs
      int n[] = {padded_signal_size}; // --- Size of the Fourier transform
      int istride = 1,
          ostride =
              1; // --- Distance between two successive input/output elements
      int idist = regularized_size,
          odist = padded_signal_size / 2 + 1; // --- Distance between batches
      int inembed[] = {
          0}; // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = {
          0}; // --- Output size with pitch (ignored for 1D transforms)
      int batch = number_signals; // --- Number of batched executions
      CufftSafeCall(cufftMakePlanMany(
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
          CUFFT_Real2Complex<real>::value, batch, &cufft_storage));

      tmp_storage.resize(cufft_storage);
      CufftSafeCall(cufftSetWorkArea(
          plan, (void *)thrust::raw_pointer_cast(tmp_storage.data())));
    }

    auto signal_ptr = thrust::raw_pointer_cast(signal.data());
    CufftSafeCall(cufftExecReal2Complex<real>(plan, (cufftReal *)signal_ptr,
                                              (cufftComplex *)signal_ptr));
    CufftSafeCall(cufftDestroy(plan));
  }
  {
    thrust::device_ptr<cufftComplex> signal_complex(
        (cufftComplex *)thrust::raw_pointer_cast(signal.data()));
    thrust::transform(
        thrust::cuda::par, signal_complex, signal_complex + signal.size() / 2,
        signal_complex, [=] __device__(cufftComplex a) {
          return cufftComplex{
              (a.x * a.x + a.y * a.y) / (real)padded_signal_size, 0};
        });
  }
  {
    // FFT both signals
    cufftHandle plan;
    {
      cufft_storage = 0;
      CufftSafeCall(cufftCreate(&plan));
      CufftSafeCall(cufftSetAutoAllocation(plan, 0));

      // Special plan for interleaved signals
      int rank = 1;                   // --- 1D FFTs
      int n[] = {padded_signal_size}; // --- Size of the Fourier transform
      int istride = 1,
          ostride =
              1; // --- Distance between two successive input/output elements
      int idist = padded_signal_size / 2 + 1,
          odist = regularized_size; // --- Distance between batches
      int inembed[] = {
          0}; // --- Input size with pitch (ignored for 1D transforms)
      int onembed[] = {
          0}; // --- Output size with pitch (ignored for 1D transforms)
      int batch = number_signals; // --- Number of batched executions
      CufftSafeCall(cufftMakePlanMany(
          plan, rank, n, inembed, istride, idist, onembed, ostride, odist,
          CUFFT_Complex2Real<real>::value, batch, &cufft_storage));
      tmp_storage.resize(cufft_storage);
      CufftSafeCall(cufftSetWorkArea(
          plan, (void *)thrust::raw_pointer_cast(tmp_storage.data())));
    }
    auto signal_ptr = thrust::raw_pointer_cast(signal.data());
    CufftSafeCall(cufftExecComplex2Real<real>(plan, (cufftComplex *)signal_ptr,
                                              (cufftReal *)signal_ptr));
    CufftSafeCall(cufftDestroy(plan));
    tmp_storage.clear();
  }
  std::vector<real> res(signal_size * number_signals);
  CudaSafeCall(cudaDeviceSynchronize());
  {
    auto signal_ptr = thrust::raw_pointer_cast(signal.data());
    for (int i = 0; i < number_signals; i++) {
      for (int j = 0; j < signal_size; j++) {
        res[i * signal_size + j] =
            signal_ptr[i * regularized_size + j] / (signal_size - j);
      }
    }
  }
  return res;
}


template std::vector<float> autocorr_cuda<float>(const std::span<float> &, int,
                                                int);
template std::vector<double> autocorr_cuda<double>(const std::span<double> &,
                                                  int, int);
} // namespace msd
