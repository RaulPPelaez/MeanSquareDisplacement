#pragma once
#include <vector>
#include <span>
#include "common.hpp"
namespace msd {
/**
 * @brief Computes the mean square displacement (MSD) of a group of trajectories
 *        (signals) in a specified number of dimensions over time.
 *
 * This function calculates the average squared displacement of particles over time,
 * which is useful in analyzing diffusion and dynamic properties in simulations or experiments.
 *
 * The input signal is assumed to be flattened and structured such that the position
 * of particle `i` at time `t` in dimension `k` is stored in:
 *
 *     signal[j + signal_size * (dimensions * i + k)]
 *
 * where:
 * - `j` is the time index (0 ≤ j < signal_size),
 * - `i` is the particle index (0 ≤ i < number_signals),
 * - `k` is the spatial dimension (0 ≤ k < dimensions).
 *
 * @tparam T Numeric data type (e.g., float, double).
 * @param signal A span over the flattened signal data.
 * @param dev The execution device (e.g., CPU, GPU).
 * @param number_signals The number of individual trajectories (particles).
 * @param signal_size The number of time points per trajectory.
 * @param dimensions The number of spatial dimensions (e.g., 2 for 2D, 3 for 3D).
 * @return A vector of type T containing the MSD values for each time lag.
 */
template <class T>
std::vector<T> mean_square_displacement(const std::span<T> &signal,
					device dev,
                                        int number_signals, int signal_size,
                                        int dimensions);


} // namespace mean_square_displacement
