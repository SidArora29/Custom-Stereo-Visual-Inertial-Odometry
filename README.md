# Custom Stereo Visual-Inertial Odometry

## Architecture
This pipeline relies on a GTSAM factor graph to fuse stereo vision and inertial data. The backend utilizes `ImuFactor` with full IMU preintegration based on the Forster et al. 2015 on-manifold formulation, tightly coupled with visual landmarks. For the visual frontend, SGBM dense disparity is used to compute stereo depth instead of sparse corner matching. The system leverages ISAM2 for incremental optimization, combined with a strict landmark age-cap algorithm to naturally bound the active memory footprint within the Bayes Tree.

## Result
The pipeline was evaluated against the millimeter-accurate Leica ground truth using the `evo` evaluation package. The results below were generated using `evo_ape` with $SE(3)$ Umeyama alignment, utilizing the exact same ground-truth source and evaluation harness as the benchmarking suite.

| Dataset | Metric | Value |
| :--- | :--- | :--- |
| **EuRoC MH_01_easy** | RMSE | 0.997 m |
| | Mean Error | 0.829 m |
| | Median Error | 0.651 m |
| | Max Error | 2.860 m |

## Key Engineering Decisions
* **SGBM over KLT stereo matching:** Swapping sparse optical flow for a dense SGBM disparity map guarantees that horizontal epipolar constraints are enforced by construction, eliminating the fragility of tracking isolated corners across two separate image streams.
* **Gravity-aligned initial pose vs identity:** By explicitly calculating the true resting gravity vector and aligning the initial pose to it, the hardware's accelerometer scale error is mathematically absorbed into the initial bias, preventing phantom thruster drift during rotation.
* **Landmark age cap vs `marginalizeLeaves`:** Because the GTSAM Python wrapper does not natively expose leaf marginalization, capping landmark lifetime forces the tracker to assign new IDs to old physical features. This orphans the old variables, allowing ISAM2's natural Bayes tree compression to bound memory usage without crashing.
* **Explicit `C(i)` camera variable vs `body_P_sensor` workaround:** Creating dedicated variables in the graph for the camera poses ensures the extrinsic calibration remains a rigid, mathematically sound glue factor, preventing the visual and inertial frames from tearing apart during aggressive maneuvers.

## Known Limitations
The trajectory exhibits a brief initialization spike at $t=0-5\text{s}$ as the IMU turn-on biases converge. Additionally, the system currently lacks a loop closure module, meaning positional drift will naturally accumulate on longer flight sequences.
