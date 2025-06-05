#pragma once
#include "defines.h"
namespace msd {
enum class device { gpu, cpu, none };
#ifdef USE_CUDA
constexpr bool gpu_mode_available = true;
#else
constexpr bool gpu_mode_available = false;
#endif
} // namespace msd
