"""Headless EGL smoke test: renders a small scene into an FBO and saves a PNG.

Scene: interpolated triangle, two depth-tested overlapping quads, a textured
checkerboard quad, rendered with 4x MSAA and resolved.
"""
import argparse
import ctypes
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from OpenGL import EGL
from OpenGL import GL
from PIL import Image

VS = """#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in vec3 col;
layout(location=2) in vec2 uv;
out vec3 v_col; out vec2 v_uv;
void main() { v_col = col; v_uv = uv; gl_Position = vec4(pos, 1.0); }
"""
FS = """#version 330 core
in vec3 v_col; in vec2 v_uv;
uniform sampler2D tex; uniform int use_tex;
out vec4 o;
void main() { o = use_tex == 1 ? texture(tex, v_uv) : vec4(v_col, 1.0); }
"""


def egl_context():
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    major, minor = EGL.EGLint(), EGL.EGLint()
    assert EGL.eglInitialize(dpy, ctypes.pointer(major), ctypes.pointer(minor))
    attribs = (EGL.EGLint * 5)(EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
                               EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT, EGL.EGL_NONE)
    cfg, n = EGL.EGLConfig(), EGL.EGLint()
    assert EGL.eglChooseConfig(dpy, attribs, ctypes.pointer(cfg), 1, ctypes.pointer(n)) and n.value
    assert EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    ctx_attr = (EGL.EGLint * 7)(EGL.EGL_CONTEXT_MAJOR_VERSION, 3, EGL.EGL_CONTEXT_MINOR_VERSION, 3,
                                EGL.EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                EGL.EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL.EGL_NONE)
    ctx = EGL.eglCreateContext(dpy, cfg, EGL.EGL_NO_CONTEXT, ctx_attr)
    assert ctx
    assert EGL.eglMakeCurrent(dpy, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, ctx)
    return dpy, ctx


def program():
    p = GL.glCreateProgram()
    for kind, src in ((GL.GL_VERTEX_SHADER, VS), (GL.GL_FRAGMENT_SHADER, FS)):
        s = GL.glCreateShader(kind)
        GL.glShaderSource(s, src)
        GL.glCompileShader(s)
        assert GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS), GL.glGetShaderInfoLog(s)
        GL.glAttachShader(p, s)
    GL.glLinkProgram(p)
    assert GL.glGetProgramiv(p, GL.GL_LINK_STATUS), GL.glGetProgramInfoLog(p)
    return p


def quad(x0, y0, x1, y1, z, c):
    v = [(x0, y0, 0, 0), (x1, y0, 1, 0), (x1, y1, 1, 1), (x0, y0, 0, 0), (x1, y1, 1, 1), (x0, y1, 0, 1)]
    return [[x, y, z, *c, u, w] for x, y, u, w in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    W = H = args.size

    dpy, _ = egl_context()
    print("renderer:", GL.glGetString(GL.GL_RENDERER).decode())

    def fbo(samples):
        fb = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fb)
        for fmt, att in ((GL.GL_RGBA8, GL.GL_COLOR_ATTACHMENT0),
                         (GL.GL_DEPTH24_STENCIL8, GL.GL_DEPTH_STENCIL_ATTACHMENT)):
            rb = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
            GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, samples, fmt, W, H)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, att, GL.GL_RENDERBUFFER, rb)
        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        return fb

    ms_fb, resolve_fb = fbo(args.samples), fbo(0)

    tri = [[-0.9, -0.9, 0.5, 1, 0, 0, 0, 0], [0.1, -0.9, 0.5, 0, 1, 0, 0, 0], [-0.4, 0.1, 0.5, 0, 0, 1, 0, 0]]
    overlap = quad(0.15, -0.9, 0.75, -0.3, 0.2, (1, 0.8, 0)) + quad(0.35, -0.7, 0.9, -0.1, -0.2, (0, 0.8, 1))
    textured = quad(-0.9, 0.25, -0.1, 0.9, 0.0, (1, 1, 1))
    data = np.array(tri + overlap + textured, dtype=np.float32)

    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
    for loc, n, off in ((0, 3, 0), (1, 3, 12), (2, 2, 24)):
        GL.glEnableVertexAttribArray(loc)
        GL.glVertexAttribPointer(loc, n, GL.GL_FLOAT, False, 32, ctypes.c_void_p(off))

    checker = (((np.indices((64, 64)).sum(0) // 8) % 2) * 255).astype(np.uint8)
    img = np.stack([checker, 255 - checker, np.full_like(checker, 128), np.full_like(checker, 255)], -1)
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, 64, 64, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

    prog = program()
    GL.glUseProgram(prog)
    use_tex = GL.glGetUniformLocation(prog, "use_tex")

    t0 = time.perf_counter()
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ms_fb)
    GL.glViewport(0, 0, W, H)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glClearColor(0.1, 0.1, 0.15, 1)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUniform1i(use_tex, 0)
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 15)
    GL.glUniform1i(use_tex, 1)
    GL.glDrawArrays(GL.GL_TRIANGLES, 15, 6)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, ms_fb)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, resolve_fb)
    GL.glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, resolve_fb)
    pixels = GL.glReadPixels(0, 0, W, H, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    print(f"first frame (incl. shader compile): {(time.perf_counter() - t0) * 1e3:.1f} ms")
    assert GL.glGetError() == GL.GL_NO_ERROR

    out = np.frombuffer(pixels, np.uint8).reshape(H, W, 4)[::-1]
    Image.fromarray(out, "RGBA").save(args.out)
    print("wrote", args.out)
    EGL.eglTerminate(dpy)


if __name__ == "__main__":
    main()
