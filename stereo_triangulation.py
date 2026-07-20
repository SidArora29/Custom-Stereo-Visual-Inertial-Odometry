import os
import cv2
import numpy as np

left_path = '/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/'
right_path = '/media/tonyox/PortableSSD/datasets/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/'

# Get sorted lists
left_files = sorted([f for f in os.listdir(left_path) if f.endswith('.png')])
right_files = sorted([f for f in os.listdir(right_path) if f.endswith('.png')])

# EuRoC Camera Intrinsics (approximated for MH_01)
focal_length = 458.6  # pixels
baseline = 0.11       # meters (distance between left and right camera)
cx = 367.2            # principal point x
cy = 248.3            # principal point y

# Read first frames
left_img = cv2.imread(os.path.join(left_path, left_files[0]), cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(os.path.join(right_path, right_files[0]), cv2.IMREAD_GRAYSCALE)

# Find corners in the LEFT image
pL = cv2.goodFeaturesToTrack(left_img, maxCorners=150, qualityLevel=0.3, minDistance=7)

# Find where those exact same corners are in the RIGHT image
pR, st, err = cv2.calcOpticalFlowPyrLK(left_img, right_img, pL, None)

# Filter good points
good_pL = pL[st == 1]
good_pR = pR[st == 1]

print(f"Raw optical flow matched {len(good_pL)} points between left and right camera.")

# =====================================================================
# OUTLIER REJECTION & TRIANGULATION
# =====================================================================
valid_3d_points = []

for i in range(len(good_pL)):
    u_L, v_L = good_pL[i].ravel()
    u_R, v_R = good_pR[i].ravel()

    # 1. Epipolar Constraint: The Y-pixel should be almost identical in both cameras
    if abs(v_L - v_R) > 15.0:
        continue # Throw it away, it's a bad match!
        
    # 2. Disparity must be positive (point must be in front of the camera)
    disparity = u_L - u_R
    if disparity <= 0.1: # Require at least a tiny bit of positive disparity
        continue # Throw it away!
        
    # 3. Calculate Valid 3D Depth
    Z = (focal_length * baseline) / disparity
    
    # Calculate 3D X and Y
    X = (u_L - cx) * Z / focal_length
    Y = (v_L - cy) * Z / focal_length
    
    valid_3d_points.append((X, Y, Z))

print(f"After Outlier Rejection, we have {len(valid_3d_points)} VALID 3D points!")

if len(valid_3d_points) > 0:
    X, Y, Z = valid_3d_points[0]
    print(f"Valid Point 0 is at 3D Coordinate (X: {X:.2f}m, Y: {Y:.2f}m, Z: {Z:.2f}m) relative to the camera lens.")