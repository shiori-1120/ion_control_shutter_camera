import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# --- 設定 ---
# フォントサイズ
BASE_FONT_SIZE = 16
LABEL_FONT_SIZE = BASE_FONT_SIZE + 2
TITLE_FONT_SIZE = BASE_FONT_SIZE + 4
SCALEBAR_FONT_SIZE = BASE_FONT_SIZE

# ファイルパス
filePath = r"C:/Users/shiori/Desktop/ion_control_shutter_camera/output/20251225/2025_1225_160504data.tif"

# トリミング範囲 [y_start:y_end, x_start:x_end]
CROP_Y_START = 100      
CROP_Y_END = 250
CROP_X_START = 150
CROP_X_END = 700
    
# ピクセルあたりの物理サイズ (µm/pixel)
PIX_TO_UM = 0.2

# --- 処理 ---
pil_img = Image.open(filePath)
np_img_original = np.array(pil_img)
np_img_cropped = np_img_original[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

fig, ax = plt.subplots(figsize=(10, 4))

im = ax.imshow(np_img_cropped * 0.2, cmap='gray')

ax.set_xlabel(r"$z$ direction (μm)", fontsize=LABEL_FONT_SIZE, style='italic')
ax.set_ylabel(r"$x$ direction (μm)", fontsize=LABEL_FONT_SIZE, style='italic')

ax.tick_params(axis='x', labelsize=BASE_FONT_SIZE)
ax.tick_params(axis='y', labelsize=BASE_FONT_SIZE)

# ax.set_title("frequency : 217.90 kHz", fontsize=TITLE_FONT_SIZE)

scalebar_length_pix = 100
scalebar_physical_um = scalebar_length_pix * PIX_TO_UM
scalebar_pos_x = 50
scalebar_pos_y = np_img_cropped.shape[0] - 50

ax.hlines(scalebar_pos_y, scalebar_pos_x, scalebar_pos_x + scalebar_length_pix, color='white', linewidth=3)
ax.text(scalebar_pos_x, scalebar_pos_y + 20, f"{scalebar_physical_um:.0f} $um", color='white', fontsize=SCALEBAR_FONT_SIZE)

cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
# cbar.set_label("photon counting in 500 msec.\n(electrons)", fontsize=LABEL_FONT_SIZE)
cbar.ax.tick_params(labelsize=BASE_FONT_SIZE)

base_name = os.path.splitext(os.path.basename(filePath))[0]
plt.savefig(f"{base_name}_cropped_output.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.tight_layout()
plt.show()
