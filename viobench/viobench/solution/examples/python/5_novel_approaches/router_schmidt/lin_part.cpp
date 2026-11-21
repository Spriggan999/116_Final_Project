// learned_lin as a router component: same source and configuration (../learned/lin), entry points renamed.
#include "viobench_abi.h"
#undef VIOBENCH_EXPORT
#define VIOBENCH_EXPORT(name)
#define abi_version lin_abi_version
#define alloc lin_alloc
#define init lin_init
#define step lin_step
extern "C" {   // C linkage for the renamed entry points (the source relies on the header's declarations)
int32_t lin_abi_version(void); void* lin_alloc(uint32_t); int32_t lin_init(const char*, uint32_t);
int32_t lin_step(double, const uint8_t*, const double*, int32_t, double*); int32_t frame_width(void);
}
#define LEARN 1
#define LEARN_LIN 1
#define LEARN_HEADER "policy_lin.h"
#include "../learned/solution.cpp"
