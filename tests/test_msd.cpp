#include "msd/msd.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <random>
using namespace msd;

TEST(MSDTest, DataOrdering) {
  // Data should be expected in the format above
  //  Use a signal that is different in each dimension and the result is
  //  expected
  std::vector<double> signal_X = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> signal_Y = {0.0, 1.0, 4.0, 9.0, 16.0};
  std::vector<double> signal(signal_X.size() + signal_Y.size());
  int nparticles = 1;
  int ntimes = signal_X.size();
  int ndim = 2;
  for (size_t i = 0; i < ntimes; ++i) {
    signal[i] = signal_X[i];          // X dimension
    signal[i + ntimes] = signal_Y[i]; // Y dimension
  }
  std::span<double> signal_span(signal);
  auto msd_result = mean_square_displacement(signal_span, device::cpu,
                                             nparticles, ntimes, ndim);
  ASSERT_EQ(msd_result.size(), ntimes * ndim);
  ASSERT_NEAR(msd_result[0], 0.0, 1e-14);                     // X dim time 0
  ASSERT_NEAR(msd_result[0 + ntimes], 0.0, 1e-14);            // Y dim time 0
  ASSERT_NEAR(msd_result[ntimes - 1], 16.0, 1e-14);           // X dim last time
  ASSERT_NEAR(msd_result[ntimes - 1 + ntimes], 256.0, 1e-14); // Y dim last time
}

TEST(MSDTest, BasicFunctionality) {
  int nparticles = 3;
  int ntimes = 3;
  int ndim = 3; // Three dimensions (X, Y, Z)
  std::vector<double> signal(nparticles * ntimes * ndim, 0);

  std::span<double> signal_span(signal);
  auto msd_result = mean_square_displacement(signal_span, device::cpu,
                                             nparticles, ntimes, ndim);
  ASSERT_EQ(msd_result.size(), ntimes * ndim); // signal_size * dimensions
  for (int i = 0; i < ntimes * ndim; ++i) {
    EXPECT_NEAR(msd_result[i], 0.0, 1e-14);
  }
}

TEST(MSDTest, RandomCloud) {
  int nparticles = 1000;
  int ntimes = 10000;
  int ndim = 1;
  std::vector<double> positions(nparticles * ndim * ntimes);
  std::mt19937 rng(422); // Fixed seed for reproducibility
  std::normal_distribution<double> dist(0.0f, 1.0f);
  for (int i = 0; i < nparticles * ndim * ntimes; ++i) {
    positions[i] = dist(rng);
  }
  std::span<double> signal_span(positions);
  auto msd_result = mean_square_displacement(signal_span, device::cpu,
                                             nparticles, ntimes, ndim);
  ASSERT_EQ(msd_result.size(), ntimes * ndim);
  // For a random cloud, the MSD should be a constant value with std*std*2 of
  // the distribution
  std::span msd_chunk(msd_result.begin() + 1, msd_result.begin() + ntimes / 2);
  double value = std::accumulate(msd_chunk.begin(), msd_chunk.end(), 0.0f) /
                 msd_chunk.size();
  double pos_sum_sq = std::inner_product(positions.begin(), positions.end(),
                                         positions.begin(), 0.0);
  double pos_std = std::sqrt(pos_sum_sq / (positions.size() - 1));
  EXPECT_NEAR(value, pos_std * pos_std * 2, 1e-2);
}
