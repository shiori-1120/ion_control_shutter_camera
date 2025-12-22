import os
import numpy as np
from typing import Optional, Dict


def build_session_dirs(timestamp: str, base_parent: Optional[str] = None) -> Dict[str, str]:
    if base_parent is None:
        base_parent = os.path.join(os.path.dirname(__file__), "..", "output")
        base_parent = os.path.normpath(base_parent)

    root = os.path.join(base_parent, timestamp)
    raw = os.path.join(root, "raw-data")
    plots = os.path.join(root, "plots")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return {"root": root, "raw": raw, "plots": plots}


def list_session_frame_paths(raw_dir: str) -> list:
    """タイムスタンプに依存せず、`raw_dir` 内の .npy を名前順で返す。"""
    names = [n for n in sorted(os.listdir(raw_dir)) if n.lower().endswith('.npy')]
    if not names:
        raise RuntimeError(f"No .npy files present in '{raw_dir}'.")
    return [os.path.join(raw_dir, n) for n in names]


def load_session_frames(raw_dir: str) -> list:
    """`list_session_frame_paths` の結果を読み込み、壊れた/空はスキップして ndarray のリストを返す。"""
    paths = list_session_frame_paths(raw_dir)
    frames: list[np.ndarray] = []
    bad: list[str] = []
    for p in paths:
        try:
            arr = np.load(p, allow_pickle=False)
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                bad.append(p)
                continue
            frames.append(arr)
        except Exception:
            bad.append(p)

    if not frames:
        raise RuntimeError(
            "No usable .npy frames after loading. "
            f"Tried {len(paths)} files in '{raw_dir}', skipped {len(bad)} bad files."
        )
    return frames
