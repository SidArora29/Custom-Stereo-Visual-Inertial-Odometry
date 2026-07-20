import gtsam
from gtsam.symbol_shorthand import X
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/home/tonyox/vio_benchmark_ws/benchmark_suite/data/MH01_imu_data.csv")

df = pd.DataFrame(data)

ax = df['a_RS_S_x [m s^-2]']
ay = df['a_RS_S_y [m s^-2]']
az = df['a_RS_S_z [m s^-2]']

df.insert(0, 'a_norm', np.sqrt(ax**2 + ay**2 + az**2))

df.plot(x='#timestamp [ns]', y='a_norm', kind='line')
plt.title('Normalized Acceleration Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Normalized Acceleration (m/s^2)')
plt.grid()
plt.show()

print(df.head())