# Franka mass-factor fusion on MI325X

Summary of the `func_factor_mass` work on Genesis 1.4.1 (editable install of `genesis-world`) for the 30k-env Franka physics benchmark on one AMD MI325X. Physics only: `gs.init(backend=gs.amdgpu, performance_mode=True)`, `show_viewer=False`, `scene.step(update_visualizer=False)`.

## Why this kernel

`rocprofv3` of the stock 30k Franka step showed `kernel_step_1` … `factor_mass` as the hottest HIP kernel: about **1.09 ms/step, ~22% of GPU time**. Collision (`narrowphase_contact0`, broadphase `traverse_valid`) and the constraint monolith were next.

`func_factor_mass` was already **one** HIP kernel. Quadrants `@qd.kernel(graph=True)` captures CUDA graphs only, so it cannot fuse launches on AMD. The cost was the algorithm Genesis selected at this env count, plus three neighboring range_fors that only exist to feed the factor.

## What was wrong at 30k envs

Genesis turns off tiled / shared-memory mass Cholesky once `n_envs` exceeds `get_gpu_core_count()` (CUs × 64 ≈ 16384 on this GPU). Franka’s mass block fits LDS, so at 30k envs the solver took the scalar path: one thread per tree/env, in-place LDL^T (`func_factor_mass_tree`).

That heuristic is aimed at CUDA, where one-thread-per-env already fills the device. On MI325X the scalar factor was the long kernel.

## What changed

Gated by `rigid_config.fuse_mass_assemble_factor`, set only when all of these hold:

- backend is `gs.amdgpu`
- tiled mass Cholesky is on and the block fits shared memory
- the scene does not require gradients

Changes, all in `genesis-world`:

1. **`rigid_solver.py`** — on AMD, treat envs as undersaturated for the mass-factor heuristic, so the shared-memory cooperative factor stays on at 30k. CUDA still uses the original env threshold.
2. **`forward_dynamics.py`** — `func_assemble_armature_factor_mass` (and the masked twin) is one `range_for` named `factor_mass`:
   - lane 0 assembles the tree mass blocks and adds armature / implicit-damping diagonal
   - block sync
   - `func_factor_mass_tree_shared` does the 32-wide LDL^T
3. **`array_class.py`** — `fuse_mass_assemble_factor` on `RigidSimStaticConfig`.

`func_compute_mass_matrix` skips `mass_mat_assemble`, `armature`, and `impint_order_1_corr` when fusion is on (`defer_assemble`). CRB and `mass_mat` (composite-inertia force) stay separate kernels; they write fields the assemble reads. `func_implicit_damping` still calls `func_factor_mass` alone for the later refactor.

## Results

Like-for-like on the **repo** sources (baseline = fusion stashed; fused = flag on). The first profiles that looked unchanged were the installed wheel in `site-packages`, not this tree. See below.

### Kernel time (`rocprofv3`, 300 steady steps)

| | baseline | fused |
|---|---:|---:|
| `factor_mass` | 1.224 ms/step | 0.823 ms/step |
| `mass_mat_assemble` + `armature` + `impint_order_1_corr` | 0.153 ms/step | gone |
| steady GPU busy | 4.586 ms/step | 4.132 ms/step |
| dispatches / step | 89 | 86 |

Most of the `factor_mass` drop is the switch from scalar LDL^T to 32-wide shared memory, not the removal of the three short kernels.

### End-to-end FPS (no profiler)

3 repeats, 50 warmup + 1000 steps, 30k envs:

| | steps/s | aggregated FPS |
|---|---:|---:|
| baseline | 228.3–228.8 | ~6.86M |
| fused | 257.9–259.1 | ~7.76M |

About **+13%**. Short windows (tens of steps) were too noisy to see this; the three 1000-step runs agree within ~0.5 steps/s.

Official Genesis Franka comment is ~43M FPS on NVIDIA at 30k envs. This change does not close that gap. After fusion, `factor_mass` is still the hottest kernel (~0.82 ms/step, ~17%), then `narrowphase_contact0` (~0.65 ms) and broadphase `traverse_valid` (~0.52 ms).

### Correctness

At 64 envs both arms use the shared cooperative factor, so the only difference is fusion. After 200 steps, `qpos` and `qvel` matched bit-for-bit.

At 30k envs the comparison also swaps scalar LDL^T for the cooperative factor (same elimination, different lane split). That arm is not bit-exact.

## How to reproduce

The venv must import the repo, not the wheel:

```bash
pip install -e /mnt/dcgpuval/younsroh/projects/physical_ai/genesis-world --no-deps
python -c "import genesis; print(genesis.__file__)"
# .../genesis-world/genesis/__init__.py
```

Scripts in this directory:

- `profile_franka_rocprofv3.sh` / `profile_franka_30k.py` — rocprofv3 via `ROCP_TOOL_LIBRARIES` (do not wrap `python` with `rocprofv3`; PyTorch’s SDK conflicts).
- `bench_franka_30k.py` — profiler-free FPS. Prints `fused=True/False`.
- `check_fused_mass.py` — dump `qpos`/`qvel` for an A/B.

Profiles used for the table:

- `out/rocprofv3/repo-baseline/`
- `out/rocprofv3/repo-fused/`

`out/rocprofv3/franka-30k-20260925-052908/` is the pre-change wheel. `054206` is an early fused capture (80 steps) before the 300-step A/B.
