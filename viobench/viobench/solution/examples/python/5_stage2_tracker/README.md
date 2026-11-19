# Stage 2: persistent two-group tracker (v2)

Classical EKF VIO for the stage-2 metric (unaligned ATE). Same filter as `../solution.cpp` (v8) with these changes, all switched
on in `CMakeLists.txt` and documented at the bottom of `params.h`:

- `TRK=2`, `TRK_SCHED=3`: features persist across windows and are tracked every 3 frames in two alternating groups; the filter
  update, the velocity prior, the covariance propagation and the feature search each run on their own frame (flat fuel).
- `TRK_DISP=1`: residuals use the IMU displacement over the tracks' own time span (no bias from the staggered groups).
- `ROT_CLONE=1`, `ROT_REFIT=0`: vision relative rotation as a filter row with an attitude clone.
- `THRUST_ONLY=1`, `MIDPOINT=1`: the simulator has no drag, so only the thrust-axis accelerometer is used; mid-interval attitude.
- `CALIB_SECONDS=0.97`, `HOVER_INIT=1`: gyro bias averaged over the whole 1 s hover (the first 0.5 s has a start-up wobble);
  tight initial velocity / tilt after a hover start.
- `SIGN_FWD=0`: no "flies nose-first" assumption. `VPRIOR_SIG=1.0`, `VPRIOR_REL=0.5`: zero-velocity prior that weakens with speed.
- `SPEED_MODE=1`, `SPEED_DEPTH=2.5`, `MIN_PAR_PX=0.5`, `N_TRACK=12`, float covariance.

Build: `viobench bundle <this folder> -o stage2_tracker_v2.wasm`.

Local results, mean unaligned ATE in metres (v8_new in brackets), max fuel 86.5k (98k):
agile medium 1.68 (2.13), agile fast 1.74 (2.22), fresh 1.43 (1.43), held-out 1.55 (1.74), public 1.20 (1.33); mean 1.55 (1.84).
