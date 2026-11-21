// Gaussian-splat map model (SPLAT=1 + relocalization): learned_lin (../../learned/lin) plus a persistent map of
// 3D Gaussian splats that is re-found by EWA-guided ZNCC search + Lucas-Kanade and used as EKF reprojection rows.
// See ../README.md. Defines below = the evaluated configuration ("splat_lite_rt": splat_lite + retuned multipliers).
#define LEARN 1
#define LEARN_LIN 1
#define LEARN_HEADER "policy_splat_lite_rt.h"   // learned_lin multipliers retuned by CMA-ES with this model on (lite_rt_cma.json)
#define SPLAT 1
#define SPL_RELOC 1          // wide relocalization of map splats (EWA window, coarse-to-fine ZNCC, LK, chi^2 gate)
#define SPL_RELOC_PF 1       // one search per frame (flat fuel)
#define SPL_RELOC_R 20       // search half-window cap (tracking px, 160x120)
#define SPL_FINE_H 1
#define SPL_NCC 0.8
#define SPL_VIEW_COS 0.7     // viewing direction within ~45 deg of the creation view
#define SPL_DRATIO 2.0       // viewing distance within x2 of the creation distance
#define SPL_T_MAP 20         // splats are created during the first 20 s (early map, little drift)
#define SPL_N 96
#define SPL_POSE_COV 1       // splat covariance = triangulation + the pose uncertainty when it was made
#define SPL_FREEZE 1         // the map is not refined afterwards
#define SPL_LINKED 0         // no rows from live tracked features (they drive the map scale down)
#define SPL_MIN_AGE 0
#define SPL_SMAX 1000
#define SPL_MAXMISS 1000
#include "../solution.cpp"
