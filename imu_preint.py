import pandas as pd
import numpy as np
import gtsam

# =====================================================================
# 1. LOAD AND PREP DATA
# =====================================================================
FILE_PATH = "/home/tonyox/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv" # Update if needed
data = pd.read_csv(FILE_PATH)
data.columns = data.columns.str.strip()

# Convert nanoseconds to seconds for easier math
data['time_sec'] = data['#timestamp [ns]'] / 1e9

# Let's isolate 10 seconds of actual flight. 
# Looking at your plot, flight starts around 46 seconds into the bag.
start_time = data['time_sec'].iloc[0] + 46.0
end_time = start_time + 10.0

flight_data = data[(data['time_sec'] >= start_time) & (data['time_sec'] <= end_time)].copy()
flight_data.reset_index(drop=True, inplace=True)

print(f"Processing {len(flight_data)} IMU measurements over 10 seconds...")

# =====================================================================
# 2. SETUP GTSAM PREINTEGRATION
# =====================================================================
# We tell GTSAM to subtract 9.81 from the Z-axis automatically
# GTSAM needs to know which way is "up". By using MakeSharedU(9.81), you are telling GTSAM: "I am using an ENU (East-North-Up) coordinate system. Gravity is pulling down, which means my IMU is experiencing a constant upward specific force of +9.81 on the Z-axis." Because GTSAM knows this, it will automatically subtract 9.81 from your Z-axis accelerometer readings before it calculates any movement.
params = gtsam.PreintegrationParams.MakeSharedU(9.81)

# Set the noise covariances for the IMU. 
# GTSAM needs 3x3 matrices. We use np.eye(3) multiplied by the variance (sigma^2).
accel_noise_sigma = 0.01
gyro_noise_sigma = 0.001
integration_noise_sigma = 1e-8

params.setAccelerometerCovariance(np.eye(3) * (accel_noise_sigma ** 2))
params.setGyroscopeCovariance(np.eye(3) * (gyro_noise_sigma ** 2))
params.setIntegrationCovariance(np.eye(3) * (integration_noise_sigma ** 2))

# =====================================================================
# 3. INITIALIZE STATE & PIM
# =====================================================================
# Create a ConstantBias object guessing zero bias for both accel and gyro.
bias_hat = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

# Create the PreintegratedImuMeasurements (PIM) object using your params and bias_hat.
pim = gtsam.PreintegratedImuMeasurements(params, bias_hat)

# Create a starting NavState (Pose + Velocity). 
# We will assume the drone starts at the origin (0,0,0) with zero velocity.
starting_navstate = gtsam.NavState(gtsam.Pose3(), np.zeros(3))

# =====================================================================
# 4. THE INTEGRATION LOOP
# =====================================================================
# We loop through every IMU measurement and feed it to the PIM.
for i in range(1, len(flight_data)):
    # Calculate dt (time difference between this row and the previous row)
    dt = flight_data['time_sec'].iloc[i] - flight_data['time_sec'].iloc[i-1]
    
    # Extract the measurements as 3-element numpy arrays
    accel = np.array([
        flight_data['a_RS_S_x [m s^-2]'].iloc[i],
        flight_data['a_RS_S_y [m s^-2]'].iloc[i],
        flight_data['a_RS_S_z [m s^-2]'].iloc[i]
    ])
    
    gyro = np.array([
        flight_data['w_RS_S_x [rad s^-1]'].iloc[i],
        flight_data['w_RS_S_y [rad s^-1]'].iloc[i],
        flight_data['w_RS_S_z [rad s^-1]'].iloc[i]
    ])
    
    # Integrate this measurement into the PIM!
    pim.integrateMeasurement(accel, gyro, dt)
    

# =====================================================================
# 5. THE PREDICTION
# =====================================================================
# Ask the PIM to predict where the drone is based on the starting state.
predicted_navstate = pim.predict(starting_navstate, bias_hat)

print("\n--- 10 SECOND DEAD RECKONING RESULT ---")
# Print the X, Y, Z translation of the predicted_navstate!
print("Final predicted translation (x, y, z) in meters:")
print(predicted_navstate.pose().translation())