#pragma once
#include <numeric>
namespace msd {
namespace detail {

template <class real> struct KahanAccumulation {
  real sum = 0;
  real correction = 0;
};

template <class real>
KahanAccumulation<real> kahan_sum(KahanAccumulation<real> accumulation,
                                  real value) {
  KahanAccumulation<real> result;
  real y = value - accumulation.correction;
  real t = accumulation.sum + y;
  result.correction = (t - accumulation.sum) - y;
  result.sum = t;
  return result;
}
} // namespace detail
template <class real, class Iterator>
real safe_accumulate(Iterator begin, Iterator end, real init = real()) {
  detail::KahanAccumulation<real> initV;
  initV.sum = init;
  auto res = std::accumulate(begin, end, initV, detail::kahan_sum<real>);
  return res.sum;
}
} // namespace mean_square_displacement
