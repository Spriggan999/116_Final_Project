# References — EECS 116 Project 3 (compute-aware VIO)

PDF downloads were blocked from Claude's sandbox, so this file holds the links. Open each one and save the PDF into `papers/`.

## Cheap visual–inertial velocity / odometry
- McGuire et al., "Efficient Optical Flow and Stereo Vision for Velocity Estimation and Obstacle Avoidance on an Autonomous Pocket Drone", RA-L 2017 — https://arxiv.org/pdf/1612.06702
  Edge-histogram flow, runs on a 168 MHz STM32F4. Blueprint for very cheap flow.
- McGuire et al., "Local Histogram Matching for Efficient Optical Flow Computation Applied to Velocity Estimation on Pocket Drones", ICRA 2016 — https://arxiv.org/pdf/1603.07644
- Bloesch et al., "Robust Visual Inertial Odometry Using a Direct EKF-Based Approach" (ROVIO), IROS 2015 — https://doi.org/10.3929/ethz-a-010566547
  Patch photometric error inside an EKF. No descriptors.
- Bouazza, Berkane, Hua, Hamel, "Observer Design for Optical Flow-Based VIO with Almost-Global Convergence", 2025 — https://arxiv.org/abs/2508.21163
  Flow + IMU → body velocity and gravity direction with convergence guarantees. **Closest to our plan.**
- Weiss, Brockers, Matthies, "Inertial Optical Flow for Throw-and-Go Micro Air Vehicles", WACV 2015 (search the title; link not verified).
- Heeger & Jepson, "Subspace methods for recovering rigid motion I", IJCV 1992 — depth-free epipolar constraint on flow (search the title).
- Suleiman et al., "Navion: A 2 mW Fully Integrated Real-Time VIO Accelerator for Nano Drones" — https://arxiv.org/pdf/1809.05780
- LEVIO: "Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices", 2026 — https://arxiv.org/pdf/2602.03294 (survey section is useful)
- Pasti, Klotz, Bellotto, Nayar, "Minimalist Visual Inertial Odometry", 2026 — https://arxiv.org/abs/2605.19990
  Four photodiodes + IMU. Shows how few visual measurements are enough.
- von Stumberg et al., "Direct Sparse VIO using Dynamic Marginalization" (VI-DSO) — https://arxiv.org/abs/1804.05625
- Forster et al., "On-Manifold Preintegration for Real-Time VIO", T-RO 2016 — https://www.ifi.uzh.ch/en/rpg/research/research_vo.html

## Contraction theory / provably stable observers (Sasha Davydov, Rice ME, and close collaborators)
- Davydov & Bullo, "Perspectives on Contractivity in Control, Optimization, and Learning", L-CSS 2024 — https://arxiv.org/abs/2404.11707  (start here)
- Davydov, Proskurnikov, Bullo, "Non-Euclidean Contraction Analysis of Continuous-Time Neural Networks", IEEE TAC 2025 — https://dblp.org/rec/journals/tac/DavydovPB25
- Jafarpour, Davydov, Proskurnikov, Bullo, "Robust Implicit Networks via Non-Euclidean Contractions", NeurIPS 2021 (search the title)
- Davydov & Bullo, "Exponential Stability of Parametric Optimization-Based Controllers via Lur'e Contractivity", L-CSS 2024 (search the title)
- Davydov et al., "Time-Varying Convex Optimization: A Contraction and Equilibrium Tracking Approach" — https://arxiv.org/abs/2305.15595
- Gokhale, Proskurnikov, Kawano, Bullo, "A Nonlinear Separation Principle via Contraction Theory" (contracting observer design via LMI), 2026 — https://arxiv.org/abs/2604.15238
- Lopez, "A Contracting Hierarchical Observer for Pose-Inertial Fusion", 2023 — https://arxiv.org/abs/2303.02777
  Orientation observer with gyro bias, then position/velocity/accel-bias observer. Same cascade as our design.
- Revay, Wang, Manchester, "Recurrent Equilibrium Networks" (§VIII: learning contracting observers) — https://arxiv.org/abs/2104.05942
- Manchester et al., "Contracting Nonlinear Observers: Convex Optimization and Learning from Data" (search the title)

## Tooling
- viobench — https://github.com/jonas-eschmann/viobench
- Wasmtime fuel ≈ 1 unit per wasm instruction — https://bytecodealliance.github.io/zulip-archive/stream/217126-wasmtime/topic/Fuel.20to.20CPU.20instructions.html
