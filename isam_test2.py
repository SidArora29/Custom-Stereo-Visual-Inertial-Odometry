#!/usr/bin/env python3
"""
Custom Stereo VIO — GTSAM Factor Graph + ISAM2
EuRoC MAV MH_01_easy

Version 5.1: Periodic ISAM2 Solver Resets to bound active memory footprint [1].
"""

import os
import csv
import pandas as pd
import numpy as np
import cv2
import gtsam
from gtsam.symbol_shorthand import X, V, B, L
import gc

def C(i):
    return gtsam.symbol('c', i)

# =====================================================================
# CONFIGURATION
# =====================================================================
RESET_INTERVAL = 400  # Re-initialize ISAM2 every 400 frames to release RAM [1]

# =====================================================================
# PATHS — adjust if your layout differs
# =====================================================================
IMU_CSV   = "/home/tonyox/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv"
CAM_CSV   = "/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data.csv"
LEFT_DIR  = "/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/"
RIGHT_DIR = "/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/"
TUM_OUT   = os.path.expanduser("~/gtsam_custom_vio/results/custom_vio_MH01.tum")

os.makedirs(os.path.dirname(TUM_OUT), exist_ok=True)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
imu_data = pd.read_csv(IMU_CSV)
imu_data.columns = imu_data.columns.str.strip()
imu_data['time_sec'] = imu_data['#timestamp [ns]'] / 1e9

cam_data = pd.read_csv(CAM_CSV)
cam_data['time_sec'] = cam_data['#timestamp [ns]'] / 1e9

left_files  = sorted([f for f in os.listdir(LEFT_DIR)  if f.endswith('.png')])
right_files = sorted([f for f in os.listdir(RIGHT_DIR) if f.endswith('.png')])

print(f"Loaded {len(cam_data)} images, {len(imu_data)} IMU readings.")
print(f"IMU:  {imu_data['time_sec'].iloc[0]:.3f} -> {imu_data['time_sec'].iloc[-1]:.3f}")
print(f"Cam:  {cam_data['time_sec'].iloc[0]:.3f} -> {cam_data['time_sec'].iloc[-1]:.3f}")

# =====================================================================
# 2. EuRoC MH_01 CALIBRATION  (from official sensor.yaml / VINS config)
# =====================================================================
IMG_W, IMG_H = 752, 480

# cam0 (left) raw intrinsics + distortion
K0 = np.array([[458.654, 0.0,     367.215],
               [0.0,     457.296, 248.375],
               [0.0,     0.0,     1.0    ]])
D0 = np.array([-0.28341811, 0.07395907, 0.00019359, 1.76187e-05])

# cam1 (right) raw intrinsics + distortion
K1 = np.array([[457.587, 0.0,     379.999],
               [0.0,     456.134, 255.238],
               [0.0,     0.0,     1.0    ]])
D1 = np.array([-0.28368366, 0.07451284, -0.00010474, -3.5555e-05])

# T_imu_cam0 = body_T_cam0 (pose of cam0 IN IMU/body frame)
T_imu_cam0 = np.array([
    [ 0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975],
    [ 0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768 ],
    [-0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949],
    [ 0.0,              0.0,              0.0,              1.0             ]
])

# T_imu_cam1 = body_T_cam1
T_imu_cam1 = np.array([
    [ 0.0125552670891, -0.999755099723,  0.0182237714554, -0.0198435579556],
    [ 0.999598781151,   0.0130119051815,  0.0251588363115,  0.0453689425024],
    [-0.0253898008918,  0.0179005838253,  0.999517347078,   0.00786212447038],
    [ 0.0,              0.0,              0.0,              1.0             ]
])

# Relative stereo geometry: cam1 relative to cam0
T_cam0_imu  = np.linalg.inv(T_imu_cam0)
T_cam0_cam1 = T_cam0_imu @ T_imu_cam1
R_01 = T_cam0_cam1[:3, :3]
t_01 = T_cam0_cam1[:3, 3]

# =====================================================================
# 3. STEREO RECTIFICATION [1]
# =====================================================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K0, D0, K1, D1, (IMG_W, IMG_H),
    R_01, t_01,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

map_l1, map_l2 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

# Rectified camera parameters for GTSAM [1]
fx_r  = float(P1[0, 0])
fy_r  = float(P1[1, 1])
cx_r  = float(P1[0, 2])
cy_r  = float(P1[1, 2])
Tx    = abs(float(P2[0, 3]))        # = fx_r * baseline
baseline = Tx / fx_r

stereo_K = gtsam.Cal3_S2Stereo(fx_r, fy_r, 0.0, cx_r, cy_r, baseline)
print(f"Rectified: fx={fx_r:.2f} fy={fy_r:.2f} cx={cx_r:.2f} cy={cy_r:.2f} b={baseline:.4f}m")

# IMU-to-rectified-cam0 extrinsics [1]
R_imu_rect = T_imu_cam0[:3, :3] @ R1.T
t_imu_rect = T_imu_cam0[:3, 3]
imu_P_cam  = gtsam.Pose3(gtsam.Rot3(R_imu_rect), gtsam.Point3(t_imu_rect))

# Tight calibration tie between X(i) and C(i).
rigid_glue_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

# =====================================================================
# 4. STATIC INITIALIZATION — gravity alignment + gyro bias seed
# =====================================================================
N_STATIC = 200      
a_mean = imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[:N_STATIC].mean().values
w_mean = imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].iloc[:N_STATIC].mean().values

print(f"Static accel mean: {a_mean}  (should be ~9.81 in one axis)")
print(f"Static gyro mean:  {w_mean}  (this becomes the initial gyro bias)")

# Align Initial Pose to the true gravity vector [1]
g_mag = np.linalg.norm(a_mean)
z_axis = a_mean / g_mag  # True UP vector in the IMU frame
x_guess = np.array([0.0, 0.0, 1.0])  # IMU Z is roughly Forward
x_axis = x_guess - np.dot(x_guess, z_axis) * z_axis
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)

R_imu_world = np.column_stack((x_axis, y_axis, z_axis))

# Align initial orientation with gravity [1]
initial_pose = gtsam.Pose3(gtsam.Rot3(R_imu_world.T), gtsam.Point3(0, 0, 0))

# Calculate initial biases (Gyro MUST be initialized to w_mean, Accel absorbs the scale offset) [1]
accel_bias = a_mean - (a_mean / g_mag * 9.81)
bias_hat = gtsam.imuBias.ConstantBias(accel_bias, w_mean)

# =====================================================================
# 5. GTSAM SETUP — ISAM2 + preintegration
# =====================================================================
graph          = gtsam.NonlinearFactorGraph()
initial_values = gtsam.Values()

isam_params = gtsam.ISAM2Params()

# Version-safe parameter binding setup [1]
try:
    isam_params.setRelinearizeThreshold(0.1)
except AttributeError:
    isam_params.relinearizeThreshold = 0.1

try:
    isam_params.setRelinearizeSkip(1)
except AttributeError:
    try:
        isam_params.relinearizeSkip = 1
    except AttributeError:
        pass

isam = gtsam.ISAM2(isam_params)

# ADIS16448 IMU noise specs (from EuRoC dataset documentation)
ACCEL_NOISE_SIGMA  = 2.0000e-3   # m/s^2 / sqrt(Hz)
GYRO_NOISE_SIGMA   = 1.6968e-4   # rad/s / sqrt(Hz)
ACCEL_BIAS_SIGMA   = 3.0000e-3   # m/s^2 * sqrt(Hz)
GYRO_BIAS_SIGMA    = 1.9393e-5   # rad/s * sqrt(Hz)

preint_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
preint_params.setAccelerometerCovariance(np.eye(3) * ACCEL_NOISE_SIGMA**2)
preint_params.setGyroscopeCovariance(np.eye(3) * GYRO_NOISE_SIGMA**2)
preint_params.setIntegrationCovariance(np.eye(3) * 1e-8)

pim = gtsam.PreintegratedImuMeasurements(preint_params, bias_hat)

# Noise models
pose_prior_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
vel_prior_noise   = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
bias_prior_noise  = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([
    np.full(3, ACCEL_BIAS_SIGMA * 10),
    np.full(3, GYRO_BIAS_SIGMA  * 10)
]))
bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([
    np.full(3, ACCEL_BIAS_SIGMA * np.sqrt(1.0 / 200)),  # integrated over ~1 IMU step
    np.full(3, GYRO_BIAS_SIGMA  * np.sqrt(1.0 / 200))
]))

# Huber robust vision noise model to handle mismatches [1]
base_vision_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
huber             = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
vision_noise      = gtsam.noiseModel.Robust.Create(huber, base_vision_noise)

# --- Anchor frame 0 (exactly once each) ---
graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, pose_prior_noise))
initial_values.insert(X(0), initial_pose)

graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), vel_prior_noise))
initial_values.insert(V(0), np.zeros(3))

graph.add(gtsam.PriorFactorConstantBias(B(0), bias_hat, bias_prior_noise))
initial_values.insert(B(0), bias_hat)

# Camera pose 0 linked to body 0 [1]
initial_cam_pose = initial_pose.compose(imu_P_cam)
initial_values.insert(C(0), initial_cam_pose)
graph.add(gtsam.BetweenFactorPose3(X(0), C(0), imu_P_cam, rigid_glue_noise))

current_navstate = gtsam.NavState(initial_pose, np.zeros(3))

# Initialize SGBM Stereo Matcher ONCE outside the loop to optimize CPU overhead [1]
stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=96,   # max disparity pixels
    blockSize=5,
    P1=8  * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# =====================================================================
# 6. VISION FRONTEND INIT on first RECTIFIED frame
# =====================================================================
# Load and pre-rectify the very first frame [1]
raw0 = cv2.imread(os.path.join(LEFT_DIR, left_files[0]), cv2.IMREAD_GRAYSCALE)
old_rect = cv2.remap(raw0, map_l1, map_l2, cv2.INTER_LINEAR)

p0 = cv2.goodFeaturesToTrack(old_rect, maxCorners=300, qualityLevel=0.005, minDistance=10)

global_lm_counter  = 0
current_lm_ids     = []
added_landmarks    = set()

if p0 is not None:
    for _ in range(len(p0)):
        current_lm_ids.append(global_lm_counter)  
        global_lm_counter += 1                     

# Trajectory records
tum_rows = []

# =====================================================================
# 7. MASTER FUSION LOOP
# =====================================================================
imu_idx = 1

for cam_idx in range(1, len(cam_data)):
    current_cam_time = cam_data['time_sec'].iloc[cam_idx]

    # --- PERIODIC ISAM2 RESET: Solves OOM/RAM growth cleanly without throwing away long-term tracks! --- [1]
    if cam_idx % RESET_INTERVAL == 0:
        print(f"\n[INFO] Performing periodic ISAM2 memory cleanup at frame {cam_idx}...")
        
        # 1. Instantiate a fresh ISAM2 solver to release all accumulated memory [1]
        isam = gtsam.ISAM2(isam_params)
        
        # 2. Clear our tracking of which landmarks are already in the graph [1]
        added_landmarks.clear()
        
        # 3. Clear graph and initial values [1]
        graph.resize(0)
        initial_values.clear()
        
        # 4. Anchor the current state (cam_idx-1) with new priors in the new solver [1]
        # This bridges the previous optimized state cleanly into the new solver instance [1]
        graph.add(gtsam.PriorFactorPose3(X(cam_idx - 1), current_navstate.pose(), pose_prior_noise))
        graph.add(gtsam.PriorFactorVector(V(cam_idx - 1), current_navstate.velocity(), vel_prior_noise))
        graph.add(gtsam.PriorFactorConstantBias(B(cam_idx - 1), bias_hat, bias_prior_noise))
        
        # Add camera pose linked to body for the new sub-graph [1]
        prev_cam_pose = current_navstate.pose().compose(imu_P_cam)
        initial_values.insert(C(cam_idx - 1), prev_cam_pose)
        graph.add(gtsam.BetweenFactorPose3(X(cam_idx - 1), C(cam_idx - 1), imu_P_cam, rigid_glue_noise))
        
        initial_values.insert(X(cam_idx - 1), current_navstate.pose())
        initial_values.insert(V(cam_idx - 1), current_navstate.velocity())
        initial_values.insert(B(cam_idx - 1), bias_hat)
        
        # Update the new ISAM2 solver with these priors to initialize it [1]
        isam.update(graph, initial_values)
        isam.update()
        
        # Clear them again to prepare for the normal Step B and C updates of the current frame [1]
        graph.resize(0)
        initial_values.clear()
        
        # Force a garbage collection of Python objects to free up RAM immediately
        gc.collect()

    # STEP A — integrate IMU up to this camera timestamp
    imu_count = 0
    while imu_idx < len(imu_data) and imu_data['time_sec'].iloc[imu_idx] <= current_cam_time:
        dt = imu_data['time_sec'].iloc[imu_idx] - imu_data['time_sec'].iloc[imu_idx - 1]
        if dt <= 0 or dt > 0.1:   # skip malformed dt values
            imu_idx += 1
            continue
        acc  = imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[imu_idx].values
        gyro = imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].iloc[imu_idx].values
        pim.integrateMeasurement(acc, gyro, dt)
        imu_idx  += 1
        imu_count += 1

    # STEP B — predict guesses and manage Jitter / Empty IMU windows cleanly [1]
    # We do NOT 'continue' if imu_count == 0 anymore, as it severs the temporal factor chain.
    if imu_count > 0:
        nav_guess  = pim.predict(current_navstate, bias_hat)
        pose_guess = nav_guess.pose()
        vel_guess  = nav_guess.velocity()

        initial_values.insert(X(cam_idx), pose_guess)
        initial_values.insert(V(cam_idx), vel_guess)
        initial_values.insert(B(cam_idx), bias_hat)

        graph.add(gtsam.ImuFactor(
            X(cam_idx - 1), V(cam_idx - 1),
            X(cam_idx),     V(cam_idx),
            B(cam_idx - 1), pim
        ))
        graph.add(gtsam.BetweenFactorConstantBias(
            B(cam_idx - 1), B(cam_idx),
            gtsam.imuBias.ConstantBias(), bias_between_noise
        ))
    else:
        # Tightly connected constant pose / velocity fallback bridge [1]
        print(f"[WARN] Sensor Jitter at frame {cam_idx}: 0 IMU packets. Bridging temporal chain.")
        pose_guess = current_navstate.pose()
        vel_guess  = current_navstate.velocity()

        initial_values.insert(X(cam_idx), pose_guess)
        initial_values.insert(V(cam_idx), vel_guess)
        initial_values.insert(B(cam_idx), bias_hat)

        graph.add(gtsam.BetweenFactorPose3(X(cam_idx - 1), X(cam_idx), gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)))
        graph.add(gtsam.PriorFactorVector(V(cam_idx), vel_guess, gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)))
        graph.add(gtsam.BetweenFactorConstantBias(
            B(cam_idx - 1), B(cam_idx),
            gtsam.imuBias.ConstantBias(), bias_between_noise
        ))

    # Camera-pose variable tied to IMU pose via calibration
    cam_pose_guess = pose_guess.compose(imu_P_cam)
    initial_values.insert(C(cam_idx), cam_pose_guess)
    graph.add(gtsam.BetweenFactorPose3(X(cam_idx), C(cam_idx), imu_P_cam, rigid_glue_noise))

    # STEP C — stereo vision on RECTIFIED images [1]
    raw_l  = cv2.imread(os.path.join(LEFT_DIR,  left_files[cam_idx]),  cv2.IMREAD_GRAYSCALE)
    raw_r  = cv2.imread(os.path.join(RIGHT_DIR, right_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    new_rect = cv2.remap(raw_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r   = cv2.remap(raw_r, map_r1, map_r2, cv2.INTER_LINEAR)

    # Compute dense disparity map using our pre-initialized stereo matcher [1]
    disp_map = stereo_matcher.compute(new_rect, rect_r).astype(np.float32) / 16.0  # [1]

    surviving_pts = []
    surviving_ids = []
    n_stereo = 0    

    if p0 is not None and len(p0) > 0:
        p1, st_l, _ = cv2.calcOpticalFlowPyrLK(old_rect, new_rect, p0, None,
                                                winSize=(21, 21), maxLevel=3)
        if p1 is not None and st_l is not None:
            good_l     = p1[st_l == 1]
            tracked_id = np.array(current_lm_ids)[(st_l == 1).flatten()]

            if len(good_l) > 0:
                # SGBM dense disparity lookup replaces right-to-left LK optical flow [1]
                for i in range(len(good_l)):
                    lm_id = int(tracked_id[i])
                    surviving_pts.append(good_l[i])
                    surviving_ids.append(lm_id)

                    u_L, v_L = good_l[i].ravel()
                    
                    # Lookup disparity at this feature's location in the dense map [1]
                    u_int = int(round(u_L))
                    v_int = int(round(v_L))

                    if not (0 <= v_int < IMG_H and 0 <= u_int < IMG_W):
                        continue

                    disp = disp_map[v_int, u_int]
                    if disp < 1.0 or np.isnan(disp):  # invalid or near-zero disparity [1]
                        continue

                    Z = Tx / disp          
                    if Z < 0.5 or Z > 20.0 or not np.isfinite(Z): 
                        continue

                    u_R = u_L - disp       # right image coordinate from disparity [1]
                    X_c = (u_L - cx_r) * Z / fx_r
                    Y_c = (v_L - cy_r) * Z / fy_r
                    
                    if not np.isfinite([X_c, Y_c]).all():
                        continue
                        
                    pt_cam   = gtsam.Point3(X_c, Y_c, Z)
                    pt_world = cam_pose_guess.transformFrom(pt_cam)

                    lm_var = L(lm_id)
                    if lm_id not in added_landmarks:
                        initial_values.insert(lm_var, pt_world)
                        added_landmarks.add(lm_id)
                        # Landmark prior restored to 5.0 (Locks down visual scale drift) [1]
                        graph.add(gtsam.PriorFactorPoint3(
                            lm_var, pt_world,
                            gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
                        ))

                    graph.add(gtsam.GenericStereoFactor3D(
                        gtsam.StereoPoint2(u_L, u_R, v_L),
                        vision_noise, C(cam_idx), lm_var, stereo_K
                    ))
                    
                    n_stereo += 1 

    # =================================================================
    # FEATURE REPLENISHMENT BLOCK (Runs frame-by-frame) [1]
    # =================================================================
    dropped_ids = set(original_lm_ids if 'original_lm_ids' in locals() else current_lm_ids) - set(surviving_ids)

    old_rect = new_rect.copy()
    current_lm_ids = surviving_ids.copy()
    p0 = (np.array(surviving_pts, dtype=np.float32).reshape(-1, 1, 2)
          if surviving_pts else np.empty((0, 1, 2), dtype=np.float32))

    if len(p0) < 150:
        new_pts = cv2.goodFeaturesToTrack(
            new_rect,
            maxCorners=200 - len(p0),
            qualityLevel=0.01,
            minDistance=10
        )
        if new_pts is not None:
            new_pts = np.float32(new_pts)
            p0 = new_pts if len(p0) == 0 else np.vstack((p0, new_pts))
            for _ in range(len(new_pts)):
                current_lm_ids.append(global_lm_counter)
                global_lm_counter += 1

    for lm_id in dropped_ids:
        added_landmarks.discard(lm_id)

    # =================================================================
    # STEP D — ISAM2 Update
    # =================================================================
    try:
        isam.update(graph, initial_values)
        isam.update()
    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        break

    result          = isam.calculateEstimate()
    current_pose    = result.atPose3(X(cam_idx))
    current_vel     = result.atVector(V(cam_idx))
    bias_hat        = result.atConstantBias(B(cam_idx))
    current_navstate = gtsam.NavState(current_pose, current_vel)

    pim = gtsam.PreintegratedImuMeasurements(preint_params, bias_hat)

    graph.resize(0)
    initial_values.clear()

    # Save to TUM
    t  = current_cam_time
    p  = current_pose.translation()
    
    # Version-safe quaternion serialization [1]
    try:
        q = current_pose.rotation().toQuaternion()
        qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
    except AttributeError:
        q_vec = current_pose.rotation().quaternion()
        qw, qx, qy, qz = q_vec[0], q_vec[1], q_vec[2], q_vec[3]

    tum_rows.append(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    # Explicitly delete the massive dense disparity array from RAM
    if 'disp_map' in locals():
        del disp_map

    if cam_idx % 100 == 0:  
        print(f"Frame {cam_idx:4d}  pos=[{p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}]  "
              f"features={len(surviving_pts):3d}  stereo={n_stereo:3d}  imu={imu_count}")
        gc.collect()  # Force RAM flush

# =====================================================================
# 8. SAVE TRAJECTORY + INSTRUCTIONS
# =====================================================================
with open(TUM_OUT, 'w') as f:
    f.write('\n'.join(tum_rows) + '\n')

print(f"\nTrajectory saved to: {TUM_OUT}")
print(f"Total frames processed: {len(tum_rows)}")
print(f"\nTo evaluate:")
print(f"  GT_TUM=~/vio_benchmark_ws/benchmark_suite/results/trajectories/MH_01_easy_gt.tum")
print(f"  evo_ape tum $GT_TUM {TUM_OUT} -a -p")