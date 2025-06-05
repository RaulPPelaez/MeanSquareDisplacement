#pragma once
#include <numeric>
namespace MeanSquareDisplacement {
namespace detail {

template <class real> struct KahanAccumulation {
  real sum = 0;
  real correction = 0;
};

template <class real>
KahanAccumulation<real> KahanSum(KahanAccumulation<real> accumulation,
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
real safeAccumulate(Iterator begin, Iterator end, real init = real()) {
  detail::KahanAccumulation<real> initV;
  initV.sum = init;
  auto res = std::accumulate(begin, end, initV, detail::KahanSum<real>);
  return res.sum;
}
} // namespace MeanSquareDisplacement
