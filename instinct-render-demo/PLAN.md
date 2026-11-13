# InstinctRender on MI355X: test plan

Physical AI simulators such as MuJoCo, Genesis, Newton and Project Chrono draw
their scenes through OpenGL. AMD Instinct MI355X (gfx950) is a compute GPU with
no fixed-function graphics hardware, so the usual AMD OpenGL stack cannot run
those renderers on it. That blocks the normal way to look at a simulated scene,
or to read RGB, depth and segmentation back into a training loop, on the same
GPU that runs the simulation.

[InstinctRender](https://github.com/AMD-BRAIN-Internal/InstinctRender/tree/instinct)
is an internal Mesa 26.2.3 fork (branch `instinct`) that adds an experimental
Gallium driver, `instinct`. It implements OpenGL 4.5 and OpenGL ES 3.2 as HIP
compute kernels, and compiles shaders at runtime with hipRTC. Applications keep
their existing OpenGL code and select the driver through EGL. The driver is not
conformant, and it accepts only gfx950, so it runs only on MI355X.

This plan is a first bring-up on a shared MI355X node: build the driver into a
private prefix, check that OpenGL actually executes on the GPU, compare images
with the `llvmpipe` CPU renderer built from the same source, and copy the
pictures to a Windows laptop. Measured results are in
[RESULTS.md](RESULTS.md).

## Hosts

| Role | Host | Notes |
|---|---|---|
| Clone, planning | `smc200x-ccs-e12-39` (MI210) | Repo cloned at `projects/physical_ai/InstinctRender` |
| Build and test | `smci355-ccs-aus-r17-22` (MI355X, ROCm 7.2, gfx950) | Shares `/mnt/dcgpuval`; build dir is local NVMe |
| Viewing | Windows laptop | Copy results with `scp`, or open them through Remote-SSH |

Access to the MI355X nodes requires a Conductor SSH key, an active reservation
and the `mi300_doc` security group.

## Steps

1. **Install build dependencies (node `-22`).** Use `apt` with Mesa's usual
   X11, Wayland, EGL and LLVM 19 development packages, plus `clang-19`,
   `libclc-19-dev`, `libllvmspirvlib-19-dev` and `llvm-spirv-19`. Create a venv
   with `meson>=1.4`, `mako`, `pyyaml`, `packaging`, `numpy`, `pillow` and
   `pyopengl`. Ubuntu's `meson` is 1.3.2, which is too old.
2. **Configure and build.** Use a private prefix so the system Mesa is not
   replaced. Build `mesa-clc` from source (`-Dmesa-clc=enabled`), because
   `-Dmesa-clc=system` has no package on Ubuntu 24.04.

   ```bash
   LLVM_CONFIG=/usr/bin/llvm-config-19 meson setup ~/build-instinct <repo> \
     --prefix=$HOME/.local/opt/mesa-instinct --libdir=lib -Dbuildtype=release \
     -Dgallium-drivers=instinct,llvmpipe -Dvulkan-drivers= \
     -Degl=enabled -Dgbm=enabled -Dglvnd=enabled -Dglvnd-vendor-name=instinct \
     -Dglx=dri -Dplatforms=x11,wayland -Dllvm=enabled -Dmesa-clc=enabled \
     -Dgallium-rusticl=false -Dgallium-va=disabled -Dbuild-tests=true
   meson compile -C ~/build-instinct && meson install -C ~/build-instinct
   ```

3. **Select the driver.** Source `scripts/instinct-env.sh`. It sets the GLVND
   EGL vendor file, the GLX vendor name, `LIBGL_ALWAYS_SOFTWARE=1`,
   `GALLIUM_DRIVER` (`instinct` by default, or `llvmpipe` for the reference) and
   `HIP_VISIBLE_DEVICES`. Pick an idle GPU with `rocm-smi` first.
4. **Confirm the driver.** Run `eglinfo -B` with `EGL_PLATFORM=surfaceless`.
   The renderer should be `instinct (HIP gfx950, experimental)`.
5. **Smoke test.** `scripts/gl_smoke.py` draws a triangle with interpolated
   colors, two overlapping depth-tested quads and a mipmapped texture, using 4x
   MSAA with a resolve. Run it with both drivers and compare with
   `scripts/compare.py`.
6. **Regression checks.** Run the Piglit tests `gl-1.0-scissor-clear`,
   `gl-1.0-scissor-depth-clear`, `depthrange-clear` and `dlist-clear` with
   `PIGLIT_PLATFORM=surfaceless_egl PIGLIT_NO_WINDOW=1`, on both drivers.
7. **Real application.** Render the MuJoCo 3.3.7 Franka Panda scene from
   `mujoco_menagerie`: RGB, depth and segmentation images, plus an animated MP4.
   Compare each output with llvmpipe.
8. **Performance.** Time steady-state frames at 640x480 and 1920x1080 after a
   warm-up. Use `INSTINCT_PROFILE=1` for per-frame driver counters.
9. **Optional, not started.** Full dEQP and Piglit suites through
   `src/gallium/drivers/instinct/ci/prepare-suite.py`, Genesis and Newton.

## Viewing from a Windows laptop

- **Files (chosen).** Copy the `results/` folder with Windows' built-in OpenSSH:

  ```powershell
  scp -r younsroh@smc200x-ccs-e12-39:/mnt/dcgpuval/younsroh/projects/physical_ai/instinct-render-demo/results .
  ```

- **Interactive (not set up).** On the node, start `Xvfb :1` and `x11vnc`, then
  tunnel with `ssh -L 5900:localhost:5900 smci355-ccs-aus-r17-22` and connect a
  VNC viewer to `localhost:5900`. Avoid `ssh -X` over a slow link.

## Risks

- The driver is experimental. Some mismatches with llvmpipe are expected, and
  unsupported operations are expected to fail explicitly.
- The MI355X nodes are shared. Check `rocm-smi` before choosing a GPU, and keep
  the reservation active.
- Installing packages with `sudo apt` changes system packages on a shared node.
  A Docker container with `--device /dev/kfd --device /dev/dri` is an alternative.
