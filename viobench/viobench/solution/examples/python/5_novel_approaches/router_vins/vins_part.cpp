// VINS as a router component (in place of the MSCKF): same source, entry points renamed, no wasm exports.
#include "viobench_abi.h"
#undef VIOBENCH_EXPORT
#define VIOBENCH_EXPORT(name)
#define abi_version msckf_abi_version
#define alloc msckf_alloc
#define init msckf_init
#define step msckf_step
#define vins_frames_since_reset msckf_frames_since_reset
#define vins_active_tracks msckf_active_tracks
#include "../vins/solution.cpp"
