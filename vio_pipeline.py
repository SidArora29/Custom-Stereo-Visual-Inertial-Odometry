import os
import pandas as pd
import numpy as np
import cv2
import gtsam
from gtsam.symbol_shorthand import X, L

# =====================================================================
# 1. LOAD SYNCHRONIZED DATA
# =====================================================================
imu_data = pd.read_csv("/home/tonyox/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv")
imu_data.columns = imu_data.columns.str.strip()
imu_data['time_sec'] = imu_data['#timestamp [ns]'] / 1e9

cam_data = pd.read_csv("/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data.csv")
cam_data['time_sec'] = cam_data['#timestamp [ns]'] / 1e9

left_path = '/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/'
right_path = '/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/'
left_files = sorted([f for f in os.listdir(left_path) if f.endswith('.png')])
right_files = sorted([f for f in os.listdir(right_path) if f.endswith('.png')])

print(f"Loaded {len(cam_data)} images and {len(imu_data)} IMU readings.")

# Camera Intrinsics & Calibration
focal_length = 458.6
baseline = 0.11
cx, cy = 367.2, 248.3

# We tell GTSAM exactly how our stereo lenses are physically built
stereo_K = gtsam.Cal3_S2Stereo(focal_length, focal_length, 0.0, cx, cy, baseline)

# =====================================================================
# 2. SETUP GTSAM GRAPH & PREINTEGRATION
# =====================================================================
graph = gtsam.NonlinearFactorGraph()
initial_values = gtsam.Values()

params = gtsam.PreintegrationParams.MakeSharedU(9.81)
params.setAccelerometerCovariance(np.eye(3) * (0.01 ** 2))
params.setGyroscopeCovariance(np.eye(3) * (0.001 ** 2))
params.setIntegrationCovariance(np.eye(3) * (1e-8 ** 2))

bias_hat = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
pim = gtsam.PreintegratedImuMeasurements(params, bias_hat)

# Anchor the very first pose
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
graph.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), prior_noise))
initial_values.insert(X(0), gtsam.Pose3())
current_navstate = gtsam.NavState(gtsam.Pose3(), np.zeros(3))

# Initialize Vision Tracking State
old_frame = cv2.imread(os.path.join(left_path, left_files[0]), cv2.IMREAD_GRAYSCALE)
p0 = cv2.goodFeaturesToTrack(old_frame, maxCorners=150, qualityLevel=0.3, minDistance=7)

# Assign a unique ID to every corner we just found so we can track them across frames!
global_landmark_counter = 0
current_landmark_ids = []
if p0 is not None:
    for _ in range(len(p0)):
        current_landmark_ids.append(global_landmark_counter)
        global_landmark_counter += 1

# =====================================================================
# 3. THE MASTER FUSION LOOP
# =====================================================================
imu_idx = 1

for cam_idx in range(1, len(cam_data)):
    current_cam_time = cam_data['time_sec'].iloc[cam_idx]
    
    # -------------------------------------------------------------
    # STEP A: INTEGRATE IMU
    # -------------------------------------------------------------
    while imu_idx < len(imu_data) and imu_data['time_sec'].iloc[imu_idx] <= current_cam_time:
        dt = imu_data['time_sec'].iloc[imu_idx] - imu_data['time_sec'].iloc[imu_idx-1]
        accel = np.array([
            imu_data['a_RS_S_x [m s^-2]'].iloc[imu_idx],
            imu_data['a_RS_S_y [m s^-2]'].iloc[imu_idx],
            imu_data['a_RS_S_z [m s^-2]'].iloc[imu_idx]
        ])
        gyro = np.array([
            imu_data['w_RS_S_x [rad s^-1]'].iloc[imu_idx],
            imu_data['w_RS_S_y [rad s^-1]'].iloc[imu_idx],
            imu_data['w_RS_S_z [rad s^-1]'].iloc[imu_idx]
        ])
        
        pim.integrateMeasurement(accel, gyro, dt)
        imu_idx += 1

    # -------------------------------------------------------------
    # STEP B: ADD IMU FACTOR TO GRAPH
    # -------------------------------------------------------------
    current_navstate = pim.predict(current_navstate, bias_hat)
    initial_values.insert(X(cam_idx), current_navstate.pose())
    
    relative_pose = gtsam.Pose3(pim.deltaRij(), pim.deltaPij())
    
    # FIX: Trust IMU rotation heavily (0.01), but completely loosen translation (10.0m) 
    # because our BetweenFactorPose3 shortcut ignores velocity momentum!
    imu_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 10.0, 10.0, 10.0]))
    graph.add(gtsam.BetweenFactorPose3(X(cam_idx-1), X(cam_idx), relative_pose, imu_noise))
    
    pim.resetIntegration()

    # -------------------------------------------------------------
    # STEP C: VISION FUSION
    # -------------------------------------------------------------
    new_frame = cv2.imread(os.path.join(left_path, left_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    right_frame = cv2.imread(os.path.join(right_path, right_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    
    surviving_p1 = []
    surviving_ids = []
    
    # 1. Track points temporally (Only if we have points to track!)
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None)
        
        if p1 is not None and st is not None:
            good_p1 = p1[st == 1]
            tracked_ids = np.array(current_landmark_ids)[(st == 1).flatten()]
            
            # 2. Stereo matching
            if len(good_p1) > 0:
                pR, st_stereo, _ = cv2.calcOpticalFlowPyrLK(new_frame, right_frame, good_p1, None)
                
                # FIX: Use Huber robust estimation to prevent single outliers from detonating the graph
                base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
                huber_estimator = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
                vision_noise = gtsam.noiseModel.Robust.Create(huber_estimator, base_noise)
                
                for i in range(len(good_p1)):
                    if st_stereo[i] == 1:
                        u_L, v_L = good_p1[i].ravel()
                        u_R, v_R = pR[i].ravel()
                        
                        # Outlier rejection
                        if abs(v_L - v_R) > 15.0: continue
                        disparity = u_L - u_R
                        if disparity <= 0.1: continue
                        
                        landmark_id = tracked_ids[i]
                        surviving_p1.append(good_p1[i])
                        surviving_ids.append(landmark_id)
                        
                        # Triangulate local 3D point
                        Z = (focal_length * baseline) / disparity
                        X_coord = (u_L - cx) * Z / focal_length
                        Y_coord = (v_L - cy) * Z / focal_length
                        local_pt = gtsam.Point3(X_coord, Y_coord, Z)
                        
                        global_pt = current_navstate.pose().transformFrom(local_pt)
                        landmark_var = L(landmark_id)
                        
                        if not initial_values.exists(landmark_var):
                            initial_values.insert(landmark_var, global_pt)
                        
                        # 3. ADD THE VISION FACTOR
                        stereo_measurement = gtsam.StereoPoint2(u_L, u_R, v_L)
                        graph.add(gtsam.GenericStereoFactor3D(
                            stereo_measurement, vision_noise, X(cam_idx), landmark_var, stereo_K
                        ))
            
    # Update state for the next camera frame
    old_frame = new_frame.copy()
    current_landmark_ids = surviving_ids
    
    # Safely convert surviving points to float32, or create empty float32 array
    if len(surviving_p1) > 0:
        p0 = np.array(surviving_p1, dtype=np.float32).reshape(-1, 1, 2)
    else:
        p0 = np.empty((0, 1, 2), dtype=np.float32)
        
    # Feature Replenishment
    if len(p0) < 100:
        new_corners = cv2.goodFeaturesToTrack(new_frame, maxCorners=150-len(p0), qualityLevel=0.3, minDistance=7)
        if new_corners is not None:
            # Ensure new_corners is float32 just in case
            new_corners = np.float32(new_corners)
            if len(p0) == 0:
                p0 = new_corners
            else:
                p0 = np.vstack((p0, new_corners))
                
            for _ in range(len(new_corners)):
                current_landmark_ids.append(global_landmark_counter)
                global_landmark_counter += 1

    if cam_idx % 100 == 0:
        print(f"Processed Frame {cam_idx}. Trailing Graph Size: {graph.size()}")

# =====================================================================
# 4. OPTIMIZE!
# =====================================================================
print(f"\nGraph construction complete. {graph.size()} total factors. Optimizing...")
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
final_result = optimizer.optimize()
print("Final Position:", final_result.atPose3(X(len(cam_data)-1)).translation())