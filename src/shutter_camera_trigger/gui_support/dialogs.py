from __future__ import annotations

from typing import Any
from tkinter import filedialog


def pick_seq_json(app: Any) -> None:
    path = filedialog.askopenfilename(
        title="Select sequence JSON",
        filetypes=[("JSON", "*.json"), ("All files", "*.*")],
    )
    if path:
        app.sw_seq_path.set(path)


def browse_dry_images(app: Any) -> None:
    path = filedialog.askdirectory(title="Select dry camera image folder")
    if path:
        app.dry_image_dir_var.set(path)
