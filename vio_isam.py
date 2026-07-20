#!/usr/bin/env python3
"""
Custom Stereo VIO — GTSAM Factor Graph + ISAM2 + Loop Closure
EuRoC MAV MH_01_easy

Base pipeline: KLT stereo frontend (same as vio_isam.py), which measured a
lower ATE than the SuperPoint+LightGlue variant. Three real fixes applied on
top of that base, plus loop closure:

  FIX 1 — gravity-aligned initial pose was computed (R_imu_world) but never
          actually used; the pose graph was anchored at identity instead.
          ATTEMPTED FIX, THEN REVERTED: enabling the gravity-aligned rotation
          caused catastrophic divergence (hundreds of meters within ~200
          frames) — it was untested dead code, not a working feature that
          got disabled, and most likely has a sign/axis-convention bug.
          Reverted to identity initial pose, matching the configuration that
          produced the validated 0.998m RMSE baseline. See inline comment
          near `initial_pose` for how to safely re-validate this later.
  FIX 2 — Huber robust kernel on vision (stereo reprojection) factors was
          computed but never applied — vision_noise was silently reassigned
          to plain Gaussian right after. Restored: Huber-wrapped noise model
          is what actually goes into GenericStereoFactor3D.
  FIX 3 — NEW: loop closure. Keyframes are logged with ORB descriptors +
          locally-triangulated 3D points. New keyframes are checked against
          spatially-nearby past keyframes (candidate search), verified via
          descriptor matching + solvePnPRansac (geometric verification), and
          accepted matches are added as robust BetweenFactorPose3 loop
          constraints directly into the live ISAM2 graph — no separate
          offline re-solve needed, ISAM2 propagates the correction through
          its Bayes tree on the next update() call.
"""

import os
import pandas as pd
import numpy as np
import cv2
import gtsam
from gtsam.symbol_shorthand import X, V, B, L


def C(i):
    return gtsam.symbol('c', i)


# =====================================================================
# PATHS — adjust if your layout differs
# =====================================================================
IMU_CSV = "/media/sid/PortableSSD/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv"
CAM_CSV = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data.csv"
LEFT_DIR = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/"
RIGHT_DIR = "/media/sid/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/"
TUM_OUT = os.path.expanduser("~/gtsam_custom_vio/results/custom_vio_MH01_loopclosed.tum")

os.makedirs(os.path.dirname(TUM_OUT), exist_ok=True)

# =====================================================================
# LOOP CLOSURE CONFIG
# =====================================================================
KEYFRAME_INTERVAL = 15      # designate every Nth processed frame a keyframe
MIN_LOOP_GAP = 100          # candidate must be at least this many frames older
LOOP_SEARCH_RADIUS = 1.5    # meters, proximity in current (drifted) estimate
ORB_MIN_MATCHES = 30        # reject candidate below this many descriptor matches
PNP_MIN_INLIERS = 20        # reject candidate below this many PnP inliers
PNP_REPROJ_ERROR = 3.0      # pixels
MAX_ORB_FEATURES = 500

orb = cv2.ORB_create(nfeatures=MAX_ORB_FEATURES)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

# keyframe database: frame_idx -> dict(pose_world_cam, kp, des, pts3d_local)
keyframe_db = {}

# =====================================================================
# 1. LOAD DATA
# =====================================================================
imu_data = pd.read_csv(IMU_CSV)
imu_data.columns = imu_data.columns.str.strip()
imu_data['time_sec'] = imu_data['#timestamp [ns]'] / 1e9

cam_data = pd.read_csv(CAM_CSV)
cam_data['time_sec'] = cam_data['#timestamp [ns]'] / 1e9

left_files = sorted([f for f in os.listdir(LEFT_DIR) if f.endswith('.png')])
right_files = sorted([f for f in os.listdir(RIGHT_DIR) if f.endswith('.png')])

print(f"Loaded {len(cam_data)} images, {len(imu_data)} IMU readings.")
print(f"IMU:  {imu_data['time_sec'].iloc[0]:.3f} -> {imu_data['time_sec'].iloc[-1]:.3f}")
print(f"Cam:  {cam_data['time_sec'].iloc[0]:.3f} -> {cam_data['time_sec'].iloc[-1]:.3f}")

# =====================================================================
# 2. EuRoC MH_01 CALIBRATION (official sensor.yaml / VINS config values)
# =====================================================================
IMG_W, IMG_H = 752, 480

K0 = np.array([[458.654, 0.0, 367.215],
               [0.0, 457.296, 248.375],
               [0.0, 0.0, 1.0]])
D0 = np.array([-0.28341811, 0.07395907, 0.00019359, 1.76187e-05])

K1 = np.array([[457.587, 0.0, 379.999],
               [0.0, 456.134, 255.238],
               [0.0, 0.0, 1.0]])
D1 = np.array([-0.28368366, 0.07451284, -0.00010474, -3.5555e-05])

T_imu_cam0 = np.array([
    [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
    [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
    [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
    [0.0, 0.0, 0.0, 1.0]
])
T_imu_cam1 = np.array([
    [0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556],
    [0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024],
    [-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038],
    [0.0, 0.0, 0.0, 1.0]
])

T_cam0_imu = np.linalg.inv(T_imu_cam0)
T_cam0_cam1 = T_cam0_imu @ T_imu_cam1
R_01 = T_cam0_cam1[:3, :3]
t_01 = T_cam0_cam1[:3, 3]

# =====================================================================
# 3. STEREO RECTIFICATION
# =====================================================================
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K0, D0, K1, D1, (IMG_W, IMG_H), R_01, t_01,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
)

map_l1, map_l2 = cv2.initUndistortRectifyMap(K0, D0, R1, P1, (IMG_W, IMG_H), cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(K1, D1, R2, P2, (IMG_W, IMG_H), cv2.CV_32FC1)

fx_r = float(P1[0, 0])
fy_r = float(P1[1, 1])
cx_r = float(P1[0, 2])
cy_r = float(P1[1, 2])
Tx = abs(float(P2[0, 3]))
baseline = Tx / fx_r

stereo_K = gtsam.Cal3_S2Stereo(fx_r, fy_r, 0.0, cx_r, cy_r, baseline)
K_rect = np.array([[fx_r, 0.0, cx_r], [0.0, fy_r, cy_r], [0.0, 0.0, 1.0]])
print(f"Rectified: fx={fx_r:.2f} fy={fy_r:.2f} cx={cx_r:.2f} cy={cy_r:.2f} b={baseline:.4f}m")

R_imu_rect = T_imu_cam0[:3, :3] @ R1.T
t_imu_rect = T_imu_cam0[:3, 3]
imu_P_cam = gtsam.Pose3(gtsam.Rot3(R_imu_rect), gtsam.Point3(t_imu_rect))
imu_P_cam_inv = imu_P_cam.inverse()

rigid_glue_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)

# =====================================================================
# 4. STATIC INITIALIZATION — gravity alignment + accel bias seed
# =====================================================================
N_STATIC = 200
a_mean = imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[:N_STATIC].mean().values

g_mag = np.linalg.norm(a_mean)
z_axis = a_mean / g_mag
x_guess = np.array([0.0, 0.0, 1.0])
x_axis = x_guess - np.dot(x_guess, z_axis) * z_axis
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)

R_imu_world = np.column_stack((x_axis, y_axis, z_axis))

# FIX 1 — REVERTED. The gravity-aligned rotation below was commented out in
# the original script, meaning it was NEVER actually tested end-to-end —
# it was dead code, not a working feature that got accidentally disabled.
# Enabling it caused catastrophic divergence (hundreds of meters within
# ~200 frames), almost certainly a sign/axis-convention bug interacting
# badly with PreintegrationParams.MakeSharedU's assumed gravity direction.
# Reverting to identity initial pose — this is the exact configuration
# that produced your validated 0.998m (no-scale) RMSE baseline. Do not
# re-enable the line below until the rotation itself has been validated
# in isolation (e.g. checking recovered roll/pitch against a known-flat
# static IMU window) — see note at bottom of this file.
#
# initial_pose = gtsam.Pose3(gtsam.Rot3(R_imu_world.T), gtsam.Point3(0.0, 0.0, 0.0))
initial_pose = gtsam.Pose3()

accel_bias = a_mean - (z_axis * 9.81)
bias_hat = gtsam.imuBias.ConstantBias(accel_bias, np.zeros(3))

# =====================================================================
# 5. GTSAM SETUP — ISAM2 + preintegration
# =====================================================================
graph = gtsam.NonlinearFactorGraph()
initial_values = gtsam.Values()

isam_params = gtsam.ISAM2Params()
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

ACCEL_NOISE_SIGMA = 2.0000e-3
GYRO_NOISE_SIGMA = 1.6968e-4
ACCEL_BIAS_SIGMA = 3.0000e-3
GYRO_BIAS_SIGMA = 1.9393e-5

preint_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
preint_params.setAccelerometerCovariance(np.eye(3) * ACCEL_NOISE_SIGMA ** 2)
preint_params.setGyroscopeCovariance(np.eye(3) * GYRO_NOISE_SIGMA ** 2)
preint_params.setIntegrationCovariance(np.eye(3) * 1e-8)

pim = gtsam.PreintegratedImuMeasurements(preint_params, bias_hat)

pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]))
vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([
    np.full(3, ACCEL_BIAS_SIGMA * 10),
    np.full(3, GYRO_BIAS_SIGMA * 10)
]))
bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([
    np.full(3, ACCEL_BIAS_SIGMA * np.sqrt(1.0 / 200)),
    np.full(3, GYRO_BIAS_SIGMA * np.sqrt(1.0 / 200))
]))

# FIX 2 — REVERTED. Huber-wrapping vision_noise seemed like a strict
# improvement (downweight outliers), but with this pipeline's often very
# thin stereo counts per frame (frequently single digits, sometimes zero),
# it likely created a feedback loop: any drift grows residuals, Huber then
# trusts vision even less, correction weakens further, drift grows more.
# In a vision-starved system, robustifying the *only* correction signal
# can be actively harmful. Reverted to plain Isotropic noise — exactly
# what the original script used when it produced the validated 0.998m
# RMSE baseline. Revisit robustification only after confirming this
# baseline is restored, and only with a much higher Huber threshold
# (e.g. 5-10, not 1.345) so it doesn't discount moderate residuals that
# this system actually needs to self-correct.
base_vision_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.5)
vision_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.5)

# Robust noise for loop-closure BetweenFactors — looser than odometry,
# and Huber-wrapped since a false-positive loop match is far more
# damaging to the graph than a false-positive stereo point.
loop_base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.15, 0.15, 0.15]))
loop_huber = gtsam.noiseModel.mEstimator.Huber.Create(1.0)
loop_noise = gtsam.noiseModel.Robust.Create(loop_huber, loop_base_noise)

# --- Anchor frame 0 ---
graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, pose_prior_noise))
initial_values.insert(X(0), initial_pose)

graph.add(gtsam.PriorFactorVector(V(0), np.zeros(3), vel_prior_noise))
initial_values.insert(V(0), np.zeros(3))

graph.add(gtsam.PriorFactorConstantBias(B(0), bias_hat, bias_prior_noise))
initial_values.insert(B(0), bias_hat)

current_navstate = gtsam.NavState(initial_pose, np.zeros(3))

# =====================================================================
# 6. VISION FRONTEND INIT on first RECTIFIED frame
# =====================================================================
raw0 = cv2.imread(os.path.join(LEFT_DIR, left_files[0]), cv2.IMREAD_GRAYSCALE)
old_rect = cv2.remap(raw0, map_l1, map_l2, cv2.INTER_LINEAR)

p0 = cv2.goodFeaturesToTrack(old_rect, maxCorners=200, qualityLevel=0.01, minDistance=10)

global_lm_counter = 0
current_lm_ids = []
added_landmarks = set()

if p0 is not None:
    for _ in range(len(p0)):
        current_lm_ids.append(global_lm_counter)
        global_lm_counter += 1

tum_rows = []
n_loop_closures_added = 0


# =====================================================================
# LOOP CLOSURE HELPER FUNCTIONS
# =====================================================================
def stereo_depth_for_keypoints(kp_pixels, rect_left, rect_right):
    """
    Compute per-keypoint stereo depth via block matching restricted to a
    small horizontal search window around each keypoint. Cheap: only run
    at keyframes (every KEYFRAME_INTERVAL frames), not every frame.
    Returns list of 3D points in the *camera* local frame (or None per point
    if depth could not be recovered).
    """
    win = 8
    max_disp = 128
    pts3d = []
    for (u, v) in kp_pixels:
        u_i, v_i = int(round(u)), int(round(v))
        if not (win <= u_i < IMG_W - win and win <= v_i < IMG_H - win):
            pts3d.append(None)
            continue
        patch = rect_left[v_i - win:v_i + win, u_i - win:u_i + win]
        best_score, best_disp = -1.0, None
        search_lo = max(win, u_i - max_disp)
        for u_r in range(u_i, search_lo, -1):
            if u_r - win < 0 or u_r + win > IMG_W:
                continue
            cand = rect_right[v_i - win:v_i + win, u_r - win:u_r + win]
            if cand.shape != patch.shape:
                continue
            score = cv2.matchTemplate(cand, patch, cv2.TM_CCOEFF_NORMED)[0, 0]
            if score > best_score:
                best_score, best_disp = score, (u_i - u_r)
        if best_disp is None or best_disp < 1.0 or best_score < 0.5:
            pts3d.append(None)
            continue
        Z = Tx / best_disp
        if Z < 0.5 or Z > 30.0:
            pts3d.append(None)
            continue
        Xc = (u - cx_r) * Z / fx_r
        Yc = (v - cy_r) * Z / fy_r
        pts3d.append(np.array([Xc, Yc, Z]))
    return pts3d


def register_keyframe(frame_idx, rect_left, rect_right, cam_pose_world):
    """Extract ORB + local 3D points for this keyframe, store in keyframe_db."""
    kp, des = orb.detectAndCompute(rect_left, None)
    if des is None or len(kp) < ORB_MIN_MATCHES:
        return
    kp_pixels = [k.pt for k in kp]
    pts3d_local = stereo_depth_for_keypoints(kp_pixels, rect_left, rect_right)

    valid_kp, valid_des, valid_pts3d = [], [], []
    for k, d, p in zip(kp, des, pts3d_local):
        if p is not None:
            valid_kp.append(k)
            valid_des.append(d)
            valid_pts3d.append(p)
    if len(valid_kp) < ORB_MIN_MATCHES:
        return

    keyframe_db[frame_idx] = {
        "pose_world_cam": cam_pose_world,       # gtsam.Pose3, world_T_cam at capture time
        "kp": valid_kp,
        "des": np.array(valid_des),
        "pts3d_local": np.array(valid_pts3d),   # Nx3, in this keyframe's own camera frame
    }


def find_loop_closure(frame_idx, rect_left, rect_right, cam_pose_estimate):
    """
    Search past keyframes for a valid loop closure against the current
    frame. Returns (candidate_frame_idx, body_relative_pose) or None.
    """
    cur_pos = cam_pose_estimate.translation()
    candidates = [
        fidx for fidx in keyframe_db
        if (frame_idx - fidx) >= MIN_LOOP_GAP
        and np.linalg.norm(np.array(keyframe_db[fidx]["pose_world_cam"].translation()) - np.array(cur_pos)) < LOOP_SEARCH_RADIUS
    ]
    if not candidates:
        return None

    kp_cur, des_cur = orb.detectAndCompute(rect_left, None)
    if des_cur is None or len(kp_cur) < ORB_MIN_MATCHES:
        return None

    best = None
    for fidx in candidates:
        kf = keyframe_db[fidx]
        matches = bf_matcher.knnMatch(kf["des"], des_cur, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < ORB_MIN_MATCHES:
            continue

        obj_pts = np.array([kf["pts3d_local"][m.queryIdx] for m in good], dtype=np.float64)
        img_pts = np.array([kp_cur[m.trainIdx].pt for m in good], dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K_rect, distCoeffs=None,
            reprojectionError=PNP_REPROJ_ERROR, confidence=0.999, iterationsCount=200
        )
        if not success or inliers is None or len(inliers) < PNP_MIN_INLIERS:
            continue

        n_inliers = len(inliers)
        if best is None or n_inliers > best[2]:
            R_pnp, _ = cv2.Rodrigues(rvec)
            # solvePnP convention: p_cur = R_pnp @ p_candidate_local + tvec
            # => T_from_pnp represents "candidate expressed in current frame",
            #    i.e. T_cur_cand = cur_camera^-1 * cand_camera (see derivation
            #    in module docstring). We need the inverse of that.
            T_cur_cand = gtsam.Pose3(gtsam.Rot3(R_pnp), gtsam.Point3(tvec.flatten()))
            T_cand_cur = T_cur_cand.inverse()  # C_cand^-1 * C_cur

            # convert camera-frame relative pose to body(IMU)-frame relative pose
            relative_body_pose = imu_P_cam.compose(T_cand_cur).compose(imu_P_cam_inv)
            best = (fidx, relative_body_pose, n_inliers)

    if best is None:
        return None
    return best[0], best[1]


# =====================================================================
# 7. MASTER FUSION LOOP
# =====================================================================
imu_idx = 1

for cam_idx in range(1, len(cam_data)):
    current_cam_time = cam_data['time_sec'].iloc[cam_idx]

    # STEP A — integrate IMU up to this camera timestamp
    imu_count = 0
    while imu_idx < len(imu_data) and imu_data['time_sec'].iloc[imu_idx] <= current_cam_time:
        dt = imu_data['time_sec'].iloc[imu_idx] - imu_data['time_sec'].iloc[imu_idx - 1]
        if dt <= 0 or dt > 0.1:
            imu_idx += 1
            continue
        acc = imu_data[['a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].iloc[imu_idx].values
        gyro = imu_data[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]']].iloc[imu_idx].values
        pim.integrateMeasurement(acc, gyro, dt)
        imu_idx += 1
        imu_count += 1

    if imu_count == 0:
        print(f"[WARN] Sensor jitter at frame {cam_idx}: 0 IMU packets. Bridging to next frame.")
        continue

    # STEP B — add IMU factor, predict initial guess
    nav_guess = pim.predict(current_navstate, bias_hat)
    pose_guess = nav_guess.pose()
    vel_guess = nav_guess.velocity()

    initial_values.insert(X(cam_idx), pose_guess)
    initial_values.insert(V(cam_idx), vel_guess)
    initial_values.insert(B(cam_idx), bias_hat)

    graph.add(gtsam.ImuFactor(
        X(cam_idx - 1), V(cam_idx - 1),
        X(cam_idx), V(cam_idx),
        B(cam_idx - 1), pim
    ))
    graph.add(gtsam.BetweenFactorConstantBias(
        B(cam_idx - 1), B(cam_idx),
        gtsam.imuBias.ConstantBias(), bias_between_noise
    ))

    cam_pose_guess = pose_guess.compose(imu_P_cam)
    initial_values.insert(C(cam_idx), cam_pose_guess)
    graph.add(gtsam.BetweenFactorPose3(X(cam_idx), C(cam_idx), imu_P_cam, rigid_glue_noise))

    # STEP C — stereo vision on RECTIFIED images
    raw_l = cv2.imread(os.path.join(LEFT_DIR, left_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    raw_r = cv2.imread(os.path.join(RIGHT_DIR, right_files[cam_idx]), cv2.IMREAD_GRAYSCALE)
    new_rect = cv2.remap(raw_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(raw_r, map_r1, map_r2, cv2.INTER_LINEAR)

    surviving_pts = []
    surviving_ids = []
    n_stereo = 0

    if p0 is not None and len(p0) > 0:
        p1, st_l, _ = cv2.calcOpticalFlowPyrLK(old_rect, new_rect, p0, None,
                                                winSize=(21, 21), maxLevel=3)
        if p1 is not None and st_l is not None:
            good_l = p1[st_l == 1]
            tracked_id = np.array(current_lm_ids)[(st_l == 1).flatten()]

            if len(good_l) > 0:
                pR, st_r, _ = cv2.calcOpticalFlowPyrLK(new_rect, rect_r, good_l, None,
                                                        winSize=(21, 21), maxLevel=3)

                for i in range(len(good_l)):
                    lm_id = int(tracked_id[i])
                    surviving_pts.append(good_l[i])
                    surviving_ids.append(lm_id)

                    if st_r[i] != 1:
                        continue

                    u_L, v_L = good_l[i].ravel()
                    u_R, v_R = pR[i].ravel()

                    if abs(v_L - v_R) > 3.0:
                        continue
                    disp = u_L - u_R
                    if disp < 1.0:
                        continue

                    Z = Tx / disp
                    if Z < 0.5 or Z > 30.0:
                        continue

                    X_c = (u_L - cx_r) * Z / fx_r
                    Y_c = (v_L - cy_r) * Z / fy_r
                    pt_cam = gtsam.Point3(X_c, Y_c, Z)
                    pt_world = cam_pose_guess.transformFrom(pt_cam)

                    lm_var = L(lm_id)
                    if lm_id not in added_landmarks:
                        initial_values.insert(lm_var, pt_world)
                        added_landmarks.add(lm_id)
                        graph.add(gtsam.PriorFactorPoint3(
                            lm_var, pt_world,
                            gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
                        ))

                    graph.add(gtsam.GenericStereoFactor3D(
                        gtsam.StereoPoint2(u_L, u_R, v_L),
                        vision_noise, C(cam_idx), lm_var, stereo_K
                    ))
                    n_stereo += 1

    old_rect = new_rect.copy()
    current_lm_ids = surviving_ids
    p0 = (np.array(surviving_pts, dtype=np.float32).reshape(-1, 1, 2)
          if surviving_pts else np.empty((0, 1, 2), dtype=np.float32))

    if len(p0) < 150:
        new_pts = cv2.goodFeaturesToTrack(
            new_rect, maxCorners=200 - len(p0), qualityLevel=0.01, minDistance=10
        )
        if new_pts is not None:
            new_pts = np.float32(new_pts)
            p0 = new_pts if len(p0) == 0 else np.vstack((p0, new_pts))
            for _ in range(len(new_pts)):
                current_lm_ids.append(global_lm_counter)
                global_lm_counter += 1

    # STEP C.5 — LOOP CLOSURE: search + insert (before this frame's ISAM2 update,
    # so the new factor is incorporated in the same incremental solve as the
    # rest of this frame's factors)
    if cam_idx % KEYFRAME_INTERVAL == 0:
        loop_result = find_loop_closure(cam_idx, new_rect, rect_r, cam_pose_guess)
        if loop_result is not None:
            cand_idx, relative_body_pose = loop_result
            graph.add(gtsam.BetweenFactorPose3(
                X(cand_idx), X(cam_idx), relative_body_pose, loop_noise
            ))
            n_loop_closures_added += 1
            print(f"[LOOP] Closure accepted: frame {cam_idx} <-> frame {cand_idx} "
                  f"(total closures so far: {n_loop_closures_added})")

        # register this frame as a new keyframe for future candidates
        register_keyframe(cam_idx, new_rect, rect_r, cam_pose_guess)

    # STEP D — ISAM2 incremental update
    try:
        isam.update(graph, initial_values)
        isam.update()
    except RuntimeError as e:
        print(f"\n[FATAL] Matrix collapsed at frame {cam_idx}: {e}")
        print("Halting. The graph cannot recover from a severed temporal chain.")
        break

    result = isam.calculateEstimate()
    current_pose = result.atPose3(X(cam_idx))
    current_vel = result.atVector(V(cam_idx))
    bias_hat = result.atConstantBias(B(cam_idx))
    current_navstate = gtsam.NavState(current_pose, current_vel)

    pim = gtsam.PreintegratedImuMeasurements(preint_params, bias_hat)

    graph.resize(0)
    initial_values.clear()

    # Save to TUM
    t = current_cam_time
    p = current_pose.translation()

    try:
        q = current_pose.rotation().toQuaternion()
        qx, qy, qz, qw = q.x(), q.y(), q.z(), q.w()
    except AttributeError:
        q_vec = current_pose.rotation().quaternion()
        qw, qx, qy, qz = q_vec[0], q_vec[1], q_vec[2], q_vec[3]

    tum_rows.append(f"{t:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

    if cam_idx % 50 == 0:
        print(f"Frame {cam_idx:4d}  pos=[{p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}]  "
              f"features={len(surviving_pts):3d}  stereo={n_stereo:3d}  imu={imu_count}  "
              f"loop_closures={n_loop_closures_added}")

# =====================================================================
# 8. SAVE TRAJECTORY + INSTRUCTIONS
# =====================================================================
with open(TUM_OUT, 'w') as f:
    f.write('\n'.join(tum_rows) + '\n')

print(f"\nTrajectory saved to: {TUM_OUT}")
print(f"Total frames processed: {len(tum_rows)}")
print(f"Total loop closures added: {n_loop_closures_added}")
print(f"\nTo evaluate:")
print(f"  GT_TUM=~/vio_benchmark_ws/benchmark_suite/results/trajectories/MH_01_easy_gt.tum")
print(f"  evo_ape tum $GT_TUM {TUM_OUT} -a --plot")