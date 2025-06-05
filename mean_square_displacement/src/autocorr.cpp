#include "msd/common.hpp"
#include "msd/autocorr.hpp"
#include <stdexcept>
#include "msd/autocorr_cpu.hpp"
#ifdef USE_CUDA
#include "msd/autocorr_cuda.cuh"
#endif
namespace msd {

template <class real>
std::vector<real> autocorr(device dev, const std::span<real> &signal,
                           int signal_size, int number_signals) {
  if (signal.size() != signal_size * number_signals) {
    throw std::runtime_error("Signal size does not match expected size.");
  }
  if (dev == device::cpu) {
    return autocorr_cpu(signal, signal_size, number_signals);
  }
#ifdef USE_CUDA
  else if (dev == device::gpu) {
    return autocorr_cuda(signal, signal_size, number_signals);

  }
#endif
  else {
    throw std::runtime_error(
        "ERROR: Unknown device for autocorrelation computation");
  }
}

template std::vector<float> autocorr<float>(device, const std::span<float> &,
                                            int, int);
template std::vector<double> autocorr<double>(device, const std::span<double> &,
                                              int, int);

} // namespace msd
