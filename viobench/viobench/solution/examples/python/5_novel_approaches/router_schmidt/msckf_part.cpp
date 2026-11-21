// The MSCKF (+ splat map) as a router component: same source, entry points renamed, no wasm exports.
#include "viobench_abi.h"
#undef VIOBENCH_EXPORT
#define VIOBENCH_EXPORT(name)
#define abi_version msckf_abi_version
#define alloc msckf_alloc
#define init msckf_init
#define step msckf_step
#include "../msckf_schmidt/solution.cpp"
