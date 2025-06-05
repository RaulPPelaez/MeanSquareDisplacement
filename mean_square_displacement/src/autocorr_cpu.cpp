#include "msd/autocorr.hpp"
#include "msd/fftwPrecisionAgnostic.h"
#include <span>
namespace msd {

template <class real>
void forward_fft_signal(std::span<real> &signal, int signal_size,
                        int number_signals, int idist) {
  using fftw_complex = fftw_complex_t<real>;
  fftw_plan_t<real> plan;
  // Special plan for interleaved signals
  int rank = 1;            // --- 1D FFTs
  int n[] = {signal_size}; // --- Size of the Fourier transform
  int istride = 1;
  // --- Distance between two successive input/output elements
  int ostride = 1;
  int odist = signal_size / 2 + 1; // --- Distance between batches
  int inembed[] = {0}; // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = {0}; // --- Output size with pitch (ignored for 1D transforms)
  int batch = number_signals; // --- Number of batched executions
  plan = fftw_plan_many_dft_r2c_prec<real>()(
      rank, n, batch, (real *)signal.data(), inembed, istride, idist,
      (fftw_complex *)signal.data(), onembed, ostride, odist, FFTW_ESTIMATE);
  msd::fftw_execute(plan);
}

template <class real>
void inverse_fft_signal(std::span<real> &signal, int signal_size,
                        int number_signals, int odist) {
  fftw_plan_t<real> plan;
  using fftw_complex = fftw_complex_t<real>;
  // Special plan for interleaved signals
  int rank = 1;            // --- 1D FFTs
  int n[] = {signal_size}; // --- Size of the Fourier transform
  int istride = 1,
      ostride = 1; // --- Distance between two successive input/output elements
  int idist = signal_size / 2 + 1;
  int inembed[] = {0}; // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = {0}; // --- Output size with pitch (ignored for 1D transforms)
  int batch = number_signals; // --- Number of batched executions
  plan = fftw_plan_many_dft_c2r_prec<real>()(
      rank, n, batch, (fftw_complex *)signal.data(), inembed, istride, idist,
      (real *)signal.data(), onembed, ostride, odist, FFTW_ESTIMATE);
  msd::fftw_execute(plan);
}

template <class real>
std::vector<real> autocorr_cpu(const std::span<real> &isignal, int signal_size,
                               int number_signals) {
  // pad with zeros
  const int padded_signal_size = signal_size * 2;
  const int regularized_size = 2 * (padded_signal_size / 2 + 1);
  std::vector<real, FFTWallocator<real>> signal(
      number_signals * regularized_size + 2, 0);
  std::span signal_span(signal);
  for (int i = 0; i < number_signals; i++) {
    for (int j = 0; j < signal_size; j++) {
      signal[j + i * regularized_size] = isignal[j + i * (signal_size)];
    }
  }
  forward_fft_signal(signal_span, padded_signal_size, number_signals,
                     regularized_size);
  using fftw_complex = fftw_complex_t<real>;
  auto *signal_complex = (fftw_complex *)(signal.data());
  for (int i = 0; i < signal.size() / 2; i++) {
    const fftw_complex a{signal_complex[i][0], signal_complex[i][1]};
    fftw_complex b;
    b[1] = 0;
    b[0] = (a[0] * a[0] + a[1] * a[1]) / (real)padded_signal_size;
    signal_complex[i][0] = b[0];
    signal_complex[i][1] = b[1];
  }
  inverse_fft_signal(signal_span, padded_signal_size, number_signals,
                     regularized_size);
  std::vector<real> res(signal_size * number_signals);
  for (int i = 0; i < number_signals; i++) {
    for (int j = 0; j < signal_size; j++) {
      res[i * signal_size + j] =
          signal[i * regularized_size + j] / (signal_size - j);
    }
  }

  return res;
}

template std::vector<float> autocorr_cpu<float>(const std::span<float> &, int,
                                                int);
template std::vector<double> autocorr_cpu<double>(const std::span<double> &,
                                                  int, int);
} // namespace msd
