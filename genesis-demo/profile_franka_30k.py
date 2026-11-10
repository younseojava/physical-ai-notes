"""Franka 30k-env step loop for rocprofv3 (physics only, no rasterizer frames)."""
import os
import time

import torch
import genesis as gs

B = int(os.environ.get("GS_N_ENVS", "30000"))
warmup = int(os.environ.get("GS_WARMUP_STEPS", "20"))
steps = int(os.environ.get("GS_PROFILE_STEPS", "50"))

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

for _ in range(warmup):
    scene.step(update_visualizer=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# rocprofv3 cannot wrap this process (PyTorch ships librocprofiler-sdk and
# rocprofiler_configure fails with error 16). Attach after warmup instead.
attach_wait = float(os.environ.get("GS_ATTACH_WAIT", "0"))
if attach_wait > 0:
    print(f"PROFILE_ATTACH_READY pid={os.getpid()} wait={attach_wait:.1f}s", flush=True)
    time.sleep(attach_wait)

t0 = time.perf_counter()
for _ in range(steps):
    scene.step(update_visualizer=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"profile_window steps={steps} n_envs={B} wall={elapsed:.3f}s "
      f"total_fps={steps * B / elapsed:,.0f} steps_per_s={steps / elapsed:.1f}", flush=True)

gs.destroy()
