"""Render a MuJoCo scene headlessly via EGL: RGB/depth/segmentation PNGs,
steady-state frame timing, and an optional animated MP4."""
import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def depth_to_u8(depth):
    valid = depth[depth < depth.max()]
    lo, hi = (valid.min(), valid.max()) if valid.size else (0.0, 1.0)
    return (255 * (1 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1))).astype(np.uint8)


def seg_to_rgb(seg):
    ids = seg[..., 0].astype(np.int64)
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(ids.max() + 2, 3), dtype=np.uint8)
    palette[0] = 0
    return palette[ids + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--video-frames", type=int, default=0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance *= 0.6
    cam.azimuth, cam.elevation = 135, -25

    with mujoco.Renderer(model, height=args.height, width=args.width) as r:
        t0 = time.perf_counter()
        r.update_scene(data, camera=cam)
        rgb = r.render()
        first = time.perf_counter() - t0
        imageio.imwrite(args.out + "_rgb.png", rgb)

        r.enable_depth_rendering()
        r.update_scene(data, camera=cam)
        imageio.imwrite(args.out + "_depth.png", depth_to_u8(r.render()))
        r.disable_depth_rendering()

        r.enable_segmentation_rendering()
        r.update_scene(data, camera=cam)
        imageio.imwrite(args.out + "_seg.png", seg_to_rgb(r.render()))
        r.disable_segmentation_rendering()

        t0 = time.perf_counter()
        for _ in range(args.frames):
            r.update_scene(data, camera=cam)
            r.render()
        dt = (time.perf_counter() - t0) / args.frames
        print(f"first frame {first * 1e3:.1f} ms; steady RGB render+readback "
              f"{dt * 1e3:.2f} ms/frame ({1 / dt:.0f} FPS) at {args.width}x{args.height}")

        if args.video_frames:
            writer = imageio.get_writer(args.out + ".mp4", fps=30, codec="libx264",
                                        quality=8, macro_block_size=8)
            home = data.ctrl.copy()
            for i in range(args.video_frames):
                t = i / 30.0
                if model.nu:
                    data.ctrl[:] = home + 0.4 * np.sin(2 * np.pi * 0.3 * t + np.arange(model.nu))
                while data.time < t:
                    mujoco.mj_step(model, data)
                cam.azimuth = 135 + 40 * np.sin(2 * np.pi * t / 8)
                r.update_scene(data, camera=cam)
                writer.append_data(r.render())
            writer.close()
            print("wrote", args.out + ".mp4")


if __name__ == "__main__":
    main()
