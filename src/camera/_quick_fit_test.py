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
for cx in (60, 160):
    for y in range(40, 80):
        for x in range(cx-3, cx+4):
            if 0 <= y < H and 0 <= x < W:
                img[y, x] += 50.0
for y in range(H):
    img[y, :] += 0.1*y
v = ip.fit_vertical_profile(img)
h = ip.fit_horizontal_profile(img)
print('vertical center,fwhm:', None if v is None else (
    float(v['center']), float(v['fwhm'])))
print('horizontal centers:', None if h is None else [
      float(c) for c in h['centers']])
print('num peaks:', None if h is None else len(h['centers']))
