#!/usr/bin/env python3
"""
Learned Stereo VIO — GTSAM Factor Graph + ISAM2 + SuperPoint/LightGlue
EuRoC MAV MH_01_easy

Includes Hybrid Architecture: SuperPoint/LightGlue frontend (1024 features) 
with strict geometric nearest-neighbor depth bounding to prevent solver divergence.
"""

import faulthandler
faulthandler.enable()

# 1. GTSAM MUST CLAIM C++ MEMORY FIRST
import gtsam
from gtsam.symbol_shorthand import X, V, B, L

# 2. PYTORCH LOADS SECOND
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd

# 3. STANDARD LIBRARIES
import os
import gc
import pandas as pd
import numpy as np
import cv2

def C(i):
    return gtsam.symbol('c', i)

# =====================================================================
# CONFIGURATION
# =====================================================================
MAX_LM_AGE = 30
# HYPOTHESIS BEING TESTED: the original value of 5 forced every feature
# track to be dropped and replaced after just 5 frames, which is likely too
# short a baseline for triangulation/parallax to disambiguate "camera moved"
# from "camera didn't move, points sit at their first-seen depth" — both
# explain a 2-5 frame observation window with similarly low residual. This
# is a classic degenerate trap in bundle adjustment with short-lived tracks.
# Raised to 30 frames to give real parallax leverage. VERIFY against the
# new diagnostic print line (n_landmarks_tracked, max_lm_age_seen) before
# trusting this — if position is still frozen with this change, the cause
# is something else and this revert should be undone.

MAX_KEYPOINTS = 512

# =====================================================================
# PATHS
# =====================================================================
IMU_CSV   = "/media/sid/PortableSSD/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv"
CAM_CSV   = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data.csv"
LEFT_DIR  = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/"
RIGHT_DIR = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/"
TUM_OUT   = os.path.expanduser("~/gtsam_custom_vio/results/learned_vio_MH01.tum")

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

# =====================================================================
# 2. CALIBRATION & RECTIFICATION 
# =====================================================================
IMG_W, IMG_H = 752, 480

K0 = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
D0 = np.array([-0.28341811, 0.07395907, 0.00019359, 1.76187e-05])
K1 = np.array([[457.587, 0.0, 379.999], [0.0, 456.134, 255.238], [0.0, 0.0, 1.0]])
D1 = np.array([-0.28368366, 0.07451284, -0.00010474, -3.5555e-05])

T_imu_cam0 = np.array([
    [ 0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975],
    [ 0.999557249008,   0.0149672133247,  0.025715529948,  -0.064676986768 ],
    [-0.0257744366974,  0.00375618835797, 0.999660727178,   0.00981073058949],
    [ 0.0,              0.0,              0.0,              1.0             ]
])
T_imu_cam1 = np.array([
    [ 0.0125552670891, -0.999755099723,  0.0182237714554, -0.0198435579556],
    [ 0.999598781151,   0.0130119051815,  0.0251588363115,  0.0453689425024],
    [-0.0253898008918,  0.0179005838253,  0.999517347078,   0.00786212447038],
    [ 0.0,              0.0,              0.0,              1.0             ]
])

T_cam0_imu  = np.linalg.inv(T_imu_cam0)
T_cam0_cam1 = T_cam0_imu @ T_imu_cam1
R_01, t_01  = T_cam0_cam1[:3, :3], T_cam0_cam1[:3, 3]

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K0, D0, K1, D1, (IMG_W, IMG_H), R_01, t_01, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
map_l1, map_l2 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

fx_r, fy_r, cx_r, cy_r = float(P1[0, 0]), float(P1[1, 1]), float(P1[0, 2]), float(P1[1, 2])
Tx = abs(float(P2[0, 3]))
baseline = Tx / fx_r

stereo_K = gtsam.Cal3_S2Stereo(fx_r, fy_r, 0.0, cx_r, cy_r, baseline)

R_imu_rect = np.ascontiguousarray(T_imu_cam0[:3, :3] @ R1.T, dtype=np.float64)
t_imu_rect = np.ascontiguousarray(T_imu_cam0[:3, 3], dtype=np.float64)
imu_P_cam  = gtsam.Pose3(gtsam.Rot3(R_imu_rect), gtsam.Point3(t_imu_rect))
rigid_glue_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

# =====================================================================
# 3. STATIC INITIALIZATION
# =====================================================================
N_STATIC = 200      
a_mean = np.ascontiguousarray(imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[:N_STATIC].mean().values, dtype=np.float64)
w_mean = np.ascontiguousarray(imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].iloc[:N_STATIC].mean().values, dtype=np.float64)

g_mag = np.linalg.norm(a_mean)
z_axis = a_mean / g_mag  
x_guess = np.array([0.0, 0.0, 1.0])  
x_axis = x_guess - np.dot(x_guess, z_axis) * z_axis
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)

R_imu_world = np.column_stack((x_axis, y_axis, z_axis))
R_imu_world_T = np.ascontiguousarray(R_imu_world.T, dtype=np.float64)
initial_pose = gtsam.Pose3(gtsam.Rot3(R_imu_world_T), gtsam.Point3(0.0, 0.0, 0.0))

accel_bias = np.ascontiguousarray(a_mean - (a_mean / g_mag * 9.81), dtype=np.float64)
bias_hat = gtsam.imuBias.ConstantBias(accel_bias, w_mean)

# =====================================================================
# 4. GTSAM SETUP
# =====================================================================
graph          = gtsam.NonlinearFactorGraph()
initial_values = gtsam.Values()

isam_params = gtsam.ISAM2Params()
try: isam_params.setRelinearizeThreshold(0.1)
except AttributeError: isam_params.relinearizeThreshold = 0.1
try: isam_params.setRelinearizeSkip(1)
except AttributeError: 
    try: isam_params.relinearizeSkip = 1
    except AttributeError: pass

isam = gtsam.ISAM2(isam_params)

# Restored official ADIS16448 datashet specs (No 10x inflation here)
ACCEL_NOISE_SIGMA  = 2.0000e-3   
GYRO_NOISE_SIGMA   = 1.6968e-4   
ACCEL_BIAS_SIGMA   = 3.0000e-3   
GYRO_BIAS_SIGMA    = 1.9393e-5   

preint_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
preint_params.setAccelerometerCovariance(np.eye(3) * ACCEL_NOISE_SIGMA**2)
preint_params.setGyroscopeCovariance(np.eye(3) * GYRO_NOISE_SIGMA**2)
preint_params.setIntegrationCovariance(np.eye(3) * 1e-8)

pim = gtsam.PreintegratedImuMeasurements(preint_params, bias_hat)

pose_prior_noise   = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
vel_prior_noise    = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
bias_prior_noise   = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([np.full(3, ACCEL_BIAS_SIGMA * 10), np.full(3, GYRO_BIAS_SIGMA * 10)]))
bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([np.full(3, ACCEL_BIAS_SIGMA * np.sqrt(1/200)), np.full(3, GYRO_BIAS_SIGMA * np.sqrt(1/200))]))

base_vision_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
huber             = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
vision_noise      = gtsam.noiseModel.Robust.Create(huber, base_vision_noise)

graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, pose_prior_noise))
initial_values.insert(X(0), initial_pose)
graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), vel_prior_noise))
initial_values.insert(V(0), np.zeros(3))
graph.add(gtsam.PriorFactorConstantBias(B(0), bias_hat, bias_prior_noise))
initial_values.insert(B(0), bias_hat)

initial_cam_pose = initial_pose.compose(imu_P_cam)
initial_values.insert(C(0), initial_cam_pose)
graph.add(gtsam.BetweenFactorPose3(X(0), C(0), imu_P_cam, rigid_glue_noise))

current_navstate = gtsam.NavState(initial_pose, np.zeros(3))

stereo_matcher = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=8 * 3 * 5**2, P2=32 * 3 * 5**2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
)

# =====================================================================
# 5. LEARNED FRONTEND INIT (SuperPoint + LightGlue)
# =====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Initializing Learned Models on {device}...")

extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
matcher   = LightGlue(features='superpoint').eval().to(device)

def image_to_tensor(img):
    return torch.from_numpy(img).float()[None, None, ...] / 255.0

raw0 = cv2.imread(os.path.join(LEFT_DIR, left_files[0]), cv2.IMREAD_GRAYSCALE)
old_rect = cv2.remap(raw0, map_l1, map_l2, cv2.INTER_LINEAR)

with torch.no_grad():
    feats0 = extractor.extract(image_to_tensor(old_rect).to(device))

global_lm_counter = 0
current_lm_ids = []
added_landmarks = set()
lm_age = {}

num_kpts = feats0['keypoints'].shape[1]
for _ in range(num_kpts):
    current_lm_ids.append(global_lm_counter)
    global_lm_counter += 1

tum_rows = []

# =====================================================================
# 6. MASTER FUSION LOOP
# =====================================================================
imu_idx = 1

for cam_idx in range(1, len(cam_data)):
    current_cam_time = cam_data['time_sec'].iloc[cam_idx]

    # STEP A — integrate IMU
    imu_count = 0
    while imu_idx < len(imu_data) and imu_data['time_sec'].iloc[imu_idx] <= current_cam_time:
        dt = imu_data['time_sec'].iloc[imu_idx] - imu_data['time_sec'].iloc[imu_idx - 1]
        if 0 < dt <= 0.1:
            acc_vals = imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[imu_idx].values
            gyro_vals = imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].iloc[imu_idx].values
            
            acc  = np.ascontiguousarray(acc_vals, dtype=np.float64)
            gyro = np.ascontiguousarray(gyro_vals, dtype=np.float64)
            
            pim.integrateMeasurement(acc, gyro, dt)
            imu_count += 1
        imu_idx += 1

    # STEP B — predict guesses
    if imu_count > 0:
        nav_guess  = pim.predict(current_navstate, bias_hat)
        pose_guess, vel_guess = nav_guess.pose(), nav_guess.velocity()
        initial_values.insert(X(cam_idx), pose_guess)
        initial_values.insert(V(cam_idx), vel_guess)
        initial_values.insert(B(cam_idx), bias_hat)
        graph.add(gtsam.ImuFactor(X(cam_idx - 1), V(cam_idx - 1), X(cam_idx), V(cam_idx), B(cam_idx - 1), pim))
        graph.add(gtsam.BetweenFactorConstantBias(B(cam_idx - 1), B(cam_idx), gtsam.imuBias.ConstantBias(), bias_between_noise))
    else:
        pose_guess, vel_guess = current_navstate.pose(), current_navstate.velocity()
        initial_values.insert(X(cam_idx), pose_guess)
        initial_values.insert(V(cam_idx), vel_guess)
        initial_values.insert(B(cam_idx), bias_hat)
        graph.add(gtsam.BetweenFactorPose3(X(cam_idx - 1), X(cam_idx), gtsam.Pose3(), gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)))
        graph.add(gtsam.PriorFactorVector(V(cam_idx), vel_guess, gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)))
        graph.add(gtsam.BetweenFactorConstantBias(B(cam_idx - 1), B(cam_idx), gtsam.imuBias.ConstantBias(), bias_between_noise))

    cam_pose_guess = pose_guess.compose(imu_P_cam)
    initial_values.insert(C(cam_idx), cam_pose_guess)
    graph.add(gtsam.BetweenFactorPose3(X(cam_idx), C(cam_idx), imu_P_cam, rigid_glue_noise))

    # STEP C — Stereo Vision & LightGlue Matching
    raw_l = cv2.imread(os.path.join(LEFT_DIR, left_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    raw_r = cv2.imread(os.path.join(RIGHT_DIR, right_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    new_rect = cv2.remap(raw_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r   = cv2.remap(raw_r, map_r1, map_r2, cv2.INTER_LINEAR)

    disp_map = stereo_matcher.compute(new_rect, rect_r).astype(np.float32) / 16.0  

    surviving_ids = []
    n_stereo = 0    

    with torch.no_grad():
        feats1 = extractor.extract(image_to_tensor(new_rect).to(device))
        matches01 = matcher({'image0': feats0, 'image1': feats1})
        feats0_clean, feats1_clean, matches01_clean = [rbd(x) for x in [feats0, feats1, matches01]]
    
    matches = matches01_clean['matches'] 
    
    # FIX 2: Reverted sub-pixel sabotage. Use purely learned feature coordinates
    pts1 = feats1_clean['keypoints'].cpu().numpy() 
    
    new_lm_ids = [-1] * pts1.shape[0] 
    
    # Process the LightGlue matches
    for i in range(matches.shape[0]):
        idx0 = matches[i, 0].item()
        idx1 = matches[i, 1].item()
        lm_id = current_lm_ids[idx0]

        lm_age[lm_id] = lm_age.get(lm_id, 0) + 1
        if lm_age[lm_id] > MAX_LM_AGE:
            continue

        u_L, v_L = pts1[idx1].ravel()
        
        # FIX 3: Reverted to nearest-neighbor lookup to prevent -1 disparity pollution
        u_int, v_int = int(round(u_L)), int(round(v_L))

        if not (0 <= v_int < IMG_H and 0 <= u_int < IMG_W): continue
        disp = disp_map[v_int, u_int]
        if disp < 1.0 or np.isnan(disp): continue

        Z = Tx / disp          
        if Z < 0.5 or Z > 20.0 or not np.isfinite(Z): continue

        surviving_ids.append(lm_id)
        new_lm_ids[idx1] = lm_id  

        u_R = u_L - disp       
        X_c = (u_L - cx_r) * Z / fx_r
        Y_c = (v_L - cy_r) * Z / fy_r
        
        if not np.isfinite([X_c, Y_c]).all(): continue
            
        pt_cam   = gtsam.Point3(X_c, Y_c, Z)
        pt_world = cam_pose_guess.transformFrom(pt_cam)

        lm_var = L(lm_id)
        if lm_id not in added_landmarks:
            initial_values.insert(lm_var, pt_world)
            added_landmarks.add(lm_id)
            graph.add(gtsam.PriorFactorPoint3(lm_var, pt_world, gtsam.noiseModel.Isotropic.Sigma(3, 5.0)))

        graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(u_L, u_R, v_L), vision_noise, C(cam_idx), lm_var, stereo_K))
        n_stereo += 1 

    # Drop orphaned features
    dropped_ids = set(current_lm_ids) - set(surviving_ids)
    for lm_id in dropped_ids:
        added_landmarks.discard(lm_id)
        lm_age.pop(lm_id, None)

    for i in range(len(new_lm_ids)):
        if new_lm_ids[i] == -1:
            new_lm_ids[i] = global_lm_counter
            global_lm_counter += 1

    feats0 = feats1 
    current_lm_ids = new_lm_ids

    # STEP D — ISAM2 Update
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
    
    try:
        q = current_pose.rotation().toQuaternion()
        qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
    except AttributeError:
        q_vec = current_pose.rotation().quaternion()
        qw, qx, qy, qz = q_vec[0], q_vec[1], q_vec[2], q_vec[3]

    tum_rows.append(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    if 'disp_map' in locals():
        del disp_map

    if cam_idx % 25 == 0:
        print(f"Frame {cam_idx:4d}  pos=[{p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}]  "
              f"stereo={n_stereo:3d}  imu={imu_count}  "
              f"bias_accel_norm={np.linalg.norm(bias_hat.accelerometer()):.4f}  "
              f"n_landmarks_tracked={len(added_landmarks)}  "
              f"max_lm_age_seen={max(lm_age.values()) if lm_age else 0}")
        if cam_idx % 100 == 0:
            gc.collect()

# =====================================================================
# 8. SAVE TRAJECTORY + INSTRUCTIONS
# =====================================================================
with open(TUM_OUT, 'w') as f:
    f.write('\n'.join(tum_rows) + '\n')

print(f"\nTrajectory saved to: {TUM_OUT}")
print(f"\nTo evaluate against your baseline:")
print(f"  evo_ape tum ~/vio_benchmark_ws/benchmark_suite/results/trajectories/MH_01_easy_gt.tum {TUM_OUT} -a -p")