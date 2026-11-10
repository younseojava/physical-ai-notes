"""Franka 30k-env physics-only FPS benchmark (no profiler, no rasterizer frames)."""
import os
import time

import torch
import genesis as gs

B = int(os.environ.get("GS_N_ENVS", "30000"))
warmup = int(os.environ.get("GS_WARMUP_STEPS", "20"))
steps = int(os.environ.get("GS_PROFILE_STEPS", "200"))

gs.init(backend=gs.amdgpu, performance_mode=True, logging_level="warning")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=False,
)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build(n_envs=B, env_spacing=(1.0, 1.0))
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], device=gs.device), (B, 1)),
)

cfg = scene.rigid_solver.rigid_config
print(
    f"config tiled={cfg.enable_tiled_cholesky_mass_matrix} fits_shared={cfg.mass_matrix_fits_shared} "
    f"fused={getattr(cfg, 'fuse_mass_assemble_factor', 'n/a')}",
    flush=True,
)

for _ in range(warmup):
    scene.step(update_visualizer=False)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(steps):
    scene.step(update_visualizer=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(
    f"RESULT steps={steps} n_envs={B} wall={elapsed:.3f}s "
    f"total_fps={steps * B / elapsed:,.0f} steps_per_s={steps / elapsed:.1f}",
    flush=True,
)

gs.destroy()
