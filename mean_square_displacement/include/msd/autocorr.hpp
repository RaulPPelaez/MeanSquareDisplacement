#pragma once
#include "common.hpp"
#include <span>
#include <vector>
namespace msd {
template <class real>
std::vector<real> autocorr(device dev, const std::span<real> &signalOr,
                           int signalSize, int Nsignals);
} // namespace msd
