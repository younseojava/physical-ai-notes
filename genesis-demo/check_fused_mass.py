"""Dump Franka trajectory state so the fused and unfused mass factor can be compared."""
import os
import sys

import torch
import genesis as gs

out = sys.argv[1]
B = 64

gs.init(backend=gs.amdgpu, performance_mode=True, logging_level="warning")
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build(n_envs=B, env_spacing=(1.0, 1.0))
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], device=gs.device), (B, 1)),
)

cfg = scene.rigid_solver.rigid_config
print(f"config tiled={cfg.enable_tiled_cholesky_mass_matrix} fused={getattr(cfg, 'fuse_mass_assemble_factor', 'n/a')}")

for _ in range(200):
    scene.step(update_visualizer=False)

torch.save(
    {"qpos": franka.get_dofs_position().cpu(), "qvel": franka.get_dofs_velocity().cpu()},
    out,
)
print("wrote", out)
gs.destroy()
