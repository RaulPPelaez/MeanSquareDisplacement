#include "msd/msd.hpp"
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <span>
namespace nb = nanobind;
using namespace nb::literals;

using pyarray = nb::ndarray<nb::device::cpu, nb::f_contig, nb::numpy>;
template <typename real>
using pyarray_t = nb::ndarray<real, nb::device::cpu, nb::f_contig, nb::numpy>;

using pyarray_c = nb::ndarray<nb::device::cpu, nb::c_contig, nb::numpy,
                              nb::shape<-1, -1, -1>>;

template <class real>
pyarray msd_dispatch_real(pyarray_t<real> signal, std::string idev,
                          size_t number_signals, size_t signal_size,
                          size_t dimensions) {
  std::span<real> signal_f(static_cast<real *>(signal.data()), signal.size());
  struct Temp {
    std::vector<real> data;
  };
  Temp *temp = new Temp;
  msd::device dev = msd::device::none;
  if (idev == "cpu") {
    dev = msd::device::cpu;
  } else if (idev == "gpu") {
    dev = msd::device::gpu;
  } else {
    throw std::invalid_argument("Unknown device: " + idev +
				". Use 'cpu' or 'gpu'.");
  }
  temp->data = msd::mean_square_displacement<real>(
      signal_f, dev, number_signals, signal_size, dimensions);
  nb::capsule deleter(temp, [](void *ptr) noexcept {
    Temp *temp = static_cast<Temp *>(ptr);
    delete temp;
  });
  pyarray_t<real> result_array(temp->data.data(), {signal_size, dimensions},
                               deleter);
  return nb::cast<pyarray>(nb::cast(result_array));
}

pyarray msd_dispatch(pyarray_c signal, std::string device) {
  // signal[j + signalSize * i] = numbers[i % dimensions];
  // j \in [0, signalSize)
  // i \in [0, number_signals * dimensions)
  size_t number_signals = signal.shape(0);
  size_t dimensions = signal.shape(1);
  size_t signal_size = signal.shape(2);
  if (signal.dtype() == nb::dtype<float>()) {
    pyarray_t<float> signal_f(signal);
    auto arr = msd_dispatch_real<float>(signal_f, device, number_signals,
                                        signal_size, dimensions);
    return arr;
  } else if (signal.dtype() == nb::dtype<double>()) {
    pyarray_t<double> signal_f(signal);
    return msd_dispatch_real<double>(signal_f, device, number_signals,
                                     signal_size, dimensions);
  } else {
    throw std::invalid_argument("Expected float32 or float64 input.");
  }
}

NB_MODULE(wrapper, m) {
  m.def("mean_square_displacement", &msd_dispatch,
	"signal"_a, "device"_a = "cpu",
        R"pbdoc(
    Compute the mean square displacement (MSD) of a set of particle trajectories over time.

    Parameters
    ----------
    signal : ndarray[float32 or float64, shape=(N, D, T)]
        A 3D array containing the trajectories of N particles over T time steps
        in D spatial dimensions. So that:

            signal[i, k, j] = position of particle i in dimension k at time j

    device : str, optional
        The device to perform computation on. Options are:
        - 'cpu' (default)
        - 'gpu'

    Returns
    -------
    ndarray
        A 2D array of shape (T, D) containing the mean square displacement for each
        time lag and each dimension, averaged for all particles.

    Notes
    -----
    The function internally flattens the data and dispatches it to a C++ backend
    that performs the MSD computation efficiently. The output shares the same
    data type as the input (`float32` or `float64`).
	 )pbdoc");
}
