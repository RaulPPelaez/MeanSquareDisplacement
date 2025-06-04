#pragma once
#include <vector>
#include <span>
#include "common.hpp"
namespace msd {
template <class T>
std::vector<T> mean_square_displacement(const std::span<T> &signal,
					device dev,
                                        int number_signals, int signal_size,
                                        int dimensions);


} // namespace mean_square_displacement
