from __future__ import annotations

from pathlib import Path
from typing import Any


def reset_spectrum_plot(fig: Any, canvas: Any) -> Any:
    """Clear the spectrum plot and return a fresh axis."""
    fig.clear()
    ax = fig.add_subplot(111)
    fig.tight_layout()
    canvas.draw()
    return ax


def update_spectrum_plot(
    ax: Any,
    canvas: Any,
    results: list[tuple[float, int, int]],
    step_idx: int,
    freq: float,
    processed: int,
    n_bright: int,
) -> None:
    if processed <= 0:
        return

    updated = False
    for i, (f, _, _) in enumerate(results):
        if abs(f - freq) < 1e-9:
            results[i] = (f, processed, n_bright)
            updated = True
            break
    if not updated:
        results.append((freq, processed, n_bright))

    xs = [f for f, _, _ in results]
    ys = [nb / n if n > 0 else 0.0 for _, n, nb in results]

    ax.clear()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("freq (Hz)")
    ax.set_ylabel("p_bright")
    ax.grid(True, alpha=0.3)
    canvas.draw()


def save_spectrum_plot(fig: Any, out_dir: Path, *, dpi: int = 120) -> None:
    fig.savefig(out_dir / "spectrum.png", dpi=int(dpi))
