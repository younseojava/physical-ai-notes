prefix=$HOME/.local/opt/mesa-instinct
export LD_LIBRARY_PATH=$prefix/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=$prefix/share/glvnd/egl_vendor.d/50_instinct.json
export __GLX_VENDOR_LIBRARY_NAME=instinct
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=${GALLIUM_DRIVER:-instinct}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
