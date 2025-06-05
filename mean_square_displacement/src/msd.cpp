#include "msd/msd.hpp"
#include "msd/autocorr.hpp"
#include "msd/common.hpp"
#include "msd/kahan.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>
namespace msd {

namespace detail {
device choose_device(int signalSize, int Nsignals, int dimensions) {
  device dev = device::cpu;
  if ((signalSize * Nsignals * dimensions > 1000000) and gpu_mode_available) {
    dev = device::gpu;
  }
  return dev;
}

template <class real>
std::vector<real> computeS1(const std::span<real> &signal, int signalSize,
                            int Nsignals, int dimensions) {
  if (signal.size() != signalSize * Nsignals * dimensions) {
    throw std::runtime_error("Signal size does not match expected size.");
  }
  std::vector<real> D(signalSize * dimensions);
  std::vector<real> S1(signalSize * dimensions);
  for (int j = 0; j < Nsignals * dimensions; j++) {
    auto signali = signal.begin() + signalSize * j;
    std::transform(signali, signali + signalSize, D.begin(),
                   [](real s) { return s * s; });
    real Q = 2.0 * safe_accumulate<real>(D.begin(), D.end());
    for (int i = 0; i < signalSize; i++) {
      if (i > 0) {
        Q -= D[i - 1] + D[signalSize - i];
      }
      S1[i + signalSize * (j % dimensions)] +=
          Q / (real(signalSize - i) * Nsignals);
    }
  }
  return std::move(S1);
}

template <class real>
std::vector<real> computeS2(const std::span<real> &signal, device dev,
                            int signalSize, int Nsignals, int dimensions) {
  // The different coordinates are simply treated as different signals
  auto S2 = autocorr<real>(dev, signal, signalSize, Nsignals * dimensions);
  std::vector<real> s2Averaged(signalSize * dimensions);
  std::vector<real> sumAux(Nsignals * dimensions);
  for (int i = 0; i < signalSize; i++) {
    for (int j = 0; j < Nsignals * dimensions; j++) {
      sumAux[j / dimensions + Nsignals * (j % dimensions)] =
          S2[i + j * signalSize] / Nsignals;
    }
    for (int j = 0; j < dimensions; j++) {
      auto first = sumAux.begin() + j * Nsignals;
      auto sum = safe_accumulate<real>(first, first + Nsignals);
      s2Averaged[i + signalSize * j] = sum;
    }
  }
  return std::move(s2Averaged);
}

template <class real>
void substract_mean(std::span<real> &signal, int Nsignals, int signalSize) {
  std::vector<real> mean(Nsignals * signalSize, 0);
  for (int i = 0; i < Nsignals; i++) {
    auto first = signal.begin() + signalSize * i;
    real mean = safe_accumulate<real>(first, first + signalSize) / signalSize;
    std::transform(first, first + signalSize, first,
                   [&](real s) { return s - mean; });
  }
}

} // namespace detail

template <class T>
std::vector<T> mean_square_displacement(const std::span<T> &isignal,
                                        device idev, int number_signals,
                                        int signal_size, int dimensions) {
  if (isignal.size() != signal_size * number_signals * dimensions) {
    throw std::runtime_error("Signal size does not match expected size.");
  }
  std::vector<T> signal(isignal.begin(), isignal.end());
  std::span<T> signal_span(signal);
  device dev = idev;
  if (dev == device::none) {
    dev = detail::choose_device(signal_size, number_signals, dimensions);
  }
  detail::substract_mean(signal_span, number_signals * dimensions, signal_size);
  auto S2 = detail::computeS2(signal_span, dev, signal_size, number_signals,
                              dimensions);
  auto S1 =
      detail::computeS1(signal_span, signal_size, number_signals, dimensions);
  std::vector<T> msd(signal_size * dimensions);
  std::transform(S1.begin(), S1.end(), S2.begin(), msd.begin(),
                 [](T S1, T S2) { return (S1 - 2.0 * S2); });
  return std::move(msd);
}

template std::vector<float> mean_square_displacement(const std::span<float> &,
                                                     device, int, int, int);
template std::vector<double> mean_square_displacement(const std::span<double> &,
                                                      device, int, int, int);
} // namespace msd
