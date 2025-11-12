import camera.initial_preparation as ip
import sys
import os
import numpy as np
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src = os.path.join(root, 'src')
if src not in sys.path:
    sys.path.insert(0, src)
H, W = 120, 240
img = np.zeros((H, W), dtype=float)
# Two real peaks
for cx in (60, 160):
    for y in range(45, 75):
        for x in range(cx-2, cx+3):
            if 0 <= y < H and 0 <= x < W:
                img[y, x] += 80.0
# Strong edge artifact at the far right border
img[:, W-2:W] += 500.0
# Vertical also add edge artifact at bottom
img[H-3:H, :] += 300.0
v = ip.fit_vertical_profile(img)
h = ip.fit_horizontal_profile(img)
print('VERT peaks:', None if v is None else v.get('peaks'))
print('VERT center:', None if v is None else float(v['center']))
print('HORIZ centers:', None if h is None else [
      float(c) for c in h['centers']])
