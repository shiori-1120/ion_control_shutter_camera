import matplotlib.pyplot as plt
import numpy as np

# Data as copied from the ndarray above
data_array = np.array([
    # Drive Current (mA)
    [0.10, 5.07, 10.09, 15.00, 20.02, 25.05, 30.05, 35.00, 40.00,
        45.00, 49.83, 55.03, 60.08, 65.05, 69.99, 75.09, 80.00],
    # Optical Power (mW)
    [0.202, -0.053, -0.076, -0.048, 0.017, 0.073, 0.143, 0.202,
        0.283, 0.411, 0.744, 1.717, 3.299, 5.119, 7.477, 9.827, 12.35]
])

current = data_array[0]  # X-axis
power = data_array[1]    # Y-axis

plt.figure(figsize=(10, 6))
plt.plot(current, power, marker='o', linestyle='-',
         color='blue', label='Power (mW)')

# Optional: Add a vertical line to estimate the threshold current (Lasing Threshold)
# The power starts increasing steeply around 50 mA


plt.title('Optical Power vs. Drive Current (L-I Curve)')
plt.xlabel('駆動電流 (Drive Current, mA)')
plt.ylabel('光強度 (Optical Power, mW)')
plt.ylim(0, 28)  # 縦軸を 0–28 mW に固定
plt.grid(True)
plt.legend()
plt.show()
