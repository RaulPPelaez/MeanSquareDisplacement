#pragma once
#include <span>
#include <vector>
namespace msd {

template <class real>
std::vector<real> autocorr_cpu(const std::span<real> &signal, int signal_size,
                               int number_signals);

} // namespace msd
