"""Compare two images; writes an amplified diff image and side-by-side strip."""
import sys

import numpy as np
from PIL import Image

a_path, b_path, out_prefix = sys.argv[1:4]
a = np.asarray(Image.open(a_path).convert("RGB")).astype(np.int16)
b = np.asarray(Image.open(b_path).convert("RGB")).astype(np.int16)
d = np.abs(a - b)
print(f"max abs diff: {d.max()}  mean: {d.mean():.4f}  "
      f"pixels differing >2: {(d.max(-1) > 2).mean() * 100:.3f}%  "
      f"PSNR: {10 * np.log10(255**2 / max((d.astype(np.float64)**2).mean(), 1e-12)):.2f} dB")
Image.fromarray(np.clip(d * 8, 0, 255).astype(np.uint8)).save(out_prefix + "_diff.png")
Image.fromarray(np.concatenate([a, b, np.clip(d * 8, 0, 255)], 1).astype(np.uint8)).save(
    out_prefix + "_side_by_side.png")
