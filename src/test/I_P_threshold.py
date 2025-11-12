import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
# data_array = np.array([
#     [0.10, 5.07, 10.09, 15.00, 20.02, 25.05, 30.05, 35.00, 40.00,
#      45.00, 49.83, 55.03, 60.08, 65.05, 69.99, 75.09, 80.00],
#     [0.202, -0.053, -0.076, -0.048, 0.017, 0.073, 0.143, 0.202,
#      0.283, 0.411, 0.744, 1.717, 3.299, 5.119, 7.477, 9.827, 12.35]
# ])

data_array = np.array([
    [
        0.014, 5.19, 10.05, 15.06, 20.20, 25.07, 30.18, 34.96, 40.31,
        45.17, 50.09, 55.10, 59.94, 64.99, 70.33, 75.01, 80.07
    ],
    [
        -0.01, 0.00, 0.00, 0.00, 0.00, 0.03, 0.09, 0.19, 2.25,
        4.61, 7.21, 9.16, 12.08, 14.32, 16.51, 18.84, 21.75
    ]
])

# data_array = np.array([
#     [
#         0.014, 5.04, 10.14, 15.02, 20.05, 25.01, 30.00, 35.19,
#         40.24, 45.05, 49.93, 55.10, 60.10, 65.00, 70.00, 75.01, 80.15
#     ],
#     [
#         0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 1.69, 4.29,
#         6.53, 9.52, 11.12, 14.67, 16.76, 19.44, 22.23, 24.04, 25.92
#     ]
# ])

# data_array = np.array([
#     [
#         0.014, 5.04, 10.08, 14.99, 19.88, 25.31, 31.52, 35.00,
#         40.83, 45.29, 49.95, 55.23, 59.48, 65.02, 70.22, 75.05,
#         80.00
#     ],
#     [
#         0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 2.62, 4.40,
#         7.60, 9.63, 12.33, 14.65, 17.13, 19.85, 22.33, 24.67,
#         27.68
#     ]
# ])

current = data_array[0]
power = data_array[1]

# --- Fit region (linear part above lasing threshold) ---
fit_region = (current > 40)  # adjust this as needed

# --- Linear fit ---
m, b = np.polyfit(current[fit_region], power[fit_region], 1)

# --- Threshold current (intersection at P = 0) ---
threshold_current = -b / m
print(f"Threshold current ≈ {threshold_current:.2f} mA")

# --- Plot ---
plt.figure(figsize=(10, 6))

# Plot data only where Power >= -0.01 mW
mask = power >= -0.01
plt.scatter(current[mask], power[mask], color='blue', label='Measured data')

# Fit line (also clipped below -0.01 mW)
fit_x = np.linspace(min(current), max(current), 200)
fit_y = m * fit_x + b
fit_y = np.where(fit_y >= -0.01, fit_y, np.nan)
plt.plot(fit_x, fit_y, '--', color='red', label='Linear fit (above threshold)')

# Threshold marker
plt.axvline(threshold_current, color='green', linestyle='--',
            label=f'Threshold ≈ {threshold_current:.2f} mA')
plt.scatter(threshold_current, 0, color='green', s=60, zorder=5)

# Labels, title, formatting
plt.title('Optical Power vs. Drive Current (L-I Curve)')
plt.xlabel('Drive Current (mA)')
plt.ylabel('Optical Power (mW)')
plt.ylim(-0.01, None)
plt.grid(True)
plt.legend()
plt.show()
