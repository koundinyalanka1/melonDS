/* GLES3 compatibility shims — force-included by build_libretro_cores.sh.
 * Stubs desktop-only OpenGL functions that glsm.c calls inside
 * #if defined(HAVE_OPENGL) blocks but which are absent from GLES3.0/3.1. */
#pragma once

/* ── Base-vertex draw calls (GLES3.2+ / GL_EXT_draw_elements_base_vertex) ──
 * Fall back to standard draw calls (basevertex ignored). */
#ifndef glDrawElementsBaseVertex
#  define glDrawElementsBaseVertex(mode,count,type,indices,bv) \
     glDrawElements(mode, count, type, indices)
#endif
#ifndef glDrawRangeElementsBaseVertex
#  define glDrawRangeElementsBaseVertex(mode,st,en,cnt,type,idx,bv) \
     glDrawRangeElements(mode, st, en, cnt, type, idx)
#endif
#ifndef glDrawElementsInstancedBaseVertex
#  define glDrawElementsInstancedBaseVertex(mode,cnt,type,idx,ic,bv) \
     glDrawElementsInstanced(mode, cnt, type, idx, ic)
#endif
#ifndef glMultiDrawElementsBaseVertex
#  define glMultiDrawElementsBaseVertex(mode,cnt,type,idx,dc,bv) ((void)0)
#endif

/* ── Multi-draw (not in GLES3 core; EXT extension only) ───────────────────*/
#ifndef glMultiDrawArrays
#  define glMultiDrawArrays(mode,first,count,dc)   ((void)0)
#endif
#ifndef glMultiDrawElements
#  define glMultiDrawElements(mode,count,type,idx,dc) ((void)0)
#endif

/* ── Base-instance draw calls (GL 4.2, not in GLES3) ─────────────────────*/
#ifndef glDrawArraysInstancedBaseInstance
#  define glDrawArraysInstancedBaseInstance(m,f,c,ic,bi) \
     glDrawArraysInstanced(m, f, c, ic)
#endif
#ifndef glDrawElementsInstancedBaseInstance
#  define glDrawElementsInstancedBaseInstance(m,c,t,i,ic,bi) \
     glDrawElementsInstanced(m, c, t, i, ic)
#endif
#ifndef glDrawElementsInstancedBaseVertexBaseInstance
#  define glDrawElementsInstancedBaseVertexBaseInstance(m,c,t,i,ic,bv,bi) \
     glDrawElementsInstanced(m, c, t, i, ic)
#endif

/* ── Transform-feedback draw calls (GL 4.x, not in GLES3) ────────────────*/
#ifndef glDrawTransformFeedback
#  define glDrawTransformFeedback(mode,id)          ((void)0)
#endif
#ifndef glDrawTransformFeedbackInstanced
#  define glDrawTransformFeedbackInstanced(m,id,ic) ((void)0)
#endif
#ifndef glDrawTransformFeedbackStream
#  define glDrawTransformFeedbackStream(m,id,s)     ((void)0)
#endif
#ifndef glDrawTransformFeedbackStreamInstanced
#  define glDrawTransformFeedbackStreamInstanced(m,id,s,ic) ((void)0)
#endif

/* ── Texture / framebuffer (GLES3.2 only) ─────────────────────────────────*/
#ifndef glTextureView
#  define glTextureView(tex,tgt,otex,fmt,ml,nl,mla,nla) ((void)0)
#endif
#ifndef glFramebufferTexture
#  define glFramebufferTexture(tgt,att,tex,lvl) \
     glFramebufferTexture2D(tgt, att, GL_TEXTURE_2D, tex, lvl)
#endif
#ifndef glTexImage2DMultisample
#  define glTexImage2DMultisample(tgt,samp,fmt,w,h,fx) ((void)0)
#endif
#ifndef glTexBuffer
#  define glTexBuffer(tgt,fmt,buf) ((void)0)
#endif

/* ── Polygon / rasteriser state (desktop only) ────────────────────────────*/
#ifndef glPolygonMode
#  define glPolygonMode(face, mode) ((void)(face),(void)(mode))
#endif
#ifndef glProvokingVertex
#  define glProvokingVertex(mode) ((void)(mode))
#endif

/* ── Per-draw-buffer blend (GLES3.2 only) ─────────────────────────────────*/
#ifndef glBlendEquationi
#  define glBlendEquationi(buf,mode)       glBlendEquation(mode)
#endif
#ifndef glBlendEquationSeparatei
#  define glBlendEquationSeparatei(buf,rgb,a) glBlendEquationSeparate(rgb,a)
#endif
#ifndef glBlendFunci
#  define glBlendFunci(buf,sfac,dfac)      glBlendFunc(sfac,dfac)
#endif
#ifndef glBlendFuncSeparatei
#  define glBlendFuncSeparatei(buf,sr,dr,sa,da) glBlendFuncSeparate(sr,dr,sa,da)
#endif
/* CRITICAL: per-draw-buffer color mask shim.
 *
 * melonDS's GPU3D_OpenGL.cpp uses MRT (color attachment 0 = RGBA color,
 * attachment 1 = polygon-attribute buffer) and issues PAIRS of
 * glColorMaski calls — one per attachment — to enable/disable writes
 * independently.  Naive shim
 *
 *     #define glColorMaski(idx, r, g, b, a) glColorMask(r, g, b, a)
 *
 * makes the SECOND call (for attachment 1) clobber the global mask the
 * first call (for attachment 0) just set.  That's a per-frame state
 * corruption — the global mask ends up reflecting attachment 1's intent
 * (often partially-FALSE), so writes to attachment 0 (the color buffer
 * we actually present) are silently masked off or partially dropped.
 *
 * GLES3.0/3.1 has no real per-attachment color masking — that's GLES3.2 /
 * GL_EXT_draw_buffers_indexed.  But melonDS's intent is "write to
 * attachment 0 fully; restrict attachment 1 partially" — so honouring
 * ONLY the idx==0 call gives correct attachment 0 behaviour and just
 * loses the (less important) attachment 1 masking, which manifests as
 * extra writes to the AttrBuffer post-process layer.  Acceptable.
 */
#ifndef glColorMaski
#  define glColorMaski(idx,r,g,b,a)  do { if ((idx) == 0) glColorMask(r,g,b,a); } while (0)
#endif
#ifndef glEnablei
#  define glEnablei(cap,idx)               glEnable(cap)
#endif
#ifndef glDisablei
#  define glDisablei(cap,idx)              glDisable(cap)
#endif

/* ── Timer queries (desktop GL 3.3 / EXT on GLES) ────────────────────────*/
#ifndef glQueryCounter
#  define glQueryCounter(id,tgt)           ((void)0)
#endif
#ifndef glGetQueryObjecti64v
#  define glGetQueryObjecti64v(id,pn,par)  ((void)0)
#endif
#ifndef glGetQueryObjectui64v
#  define glGetQueryObjectui64v(id,pn,par) ((void)0)
#endif

/* ── Immutable buffer storage (GL 4.4 / GL_EXT_buffer_storage) ───────────
 * Fall back to glBufferData with DYNAMIC_DRAW usage. */
#ifndef glBufferStorage
#  define glBufferStorage(tgt,sz,data,flags) \
     glBufferData(tgt, sz, data, GL_DYNAMIC_DRAW)
#endif

/* ── Buffer-clearing utilities (GL 4.3, not in GLES3) ────────────────────*/
#ifndef glClearBufferData
#  define glClearBufferData(tgt,ifmt,fmt,type,data) ((void)0)
#endif
#ifndef glClearBufferSubData
#  define glClearBufferSubData(tgt,ifmt,ofs,sz,fmt,t,d) ((void)0)
#endif
#ifndef glInvalidateBufferData
#  define glInvalidateBufferData(buf) ((void)0)
#endif
#ifndef glInvalidateBufferSubData
#  define glInvalidateBufferSubData(buf,ofs,len) ((void)0)
#endif

/* ── Compute (GLES3.1, not GLES3.0) ──────────────────────────────────────*/
#ifndef glDispatchCompute
#  define glDispatchCompute(x,y,z) ((void)0)
#endif
#ifndef glMemoryBarrier
#  define glMemoryBarrier(bits) ((void)0)
#endif
#ifndef glBindImageTexture
#  define glBindImageTexture(u,t,lvl,lyr,layer,ac,fmt) ((void)0)
#endif

/* ── Clip control (GL 4.5, not in GLES3) ────────────────────────────────*/
#ifndef glClipControl
#  define glClipControl(origin, depth) ((void)0)
#endif

/* ── Buffer readback (not in GLES3 core; use glMapBufferRange instead) ────*/
#ifndef glGetBufferSubData
#  define glGetBufferSubData(tgt,ofs,sz,data) ((void)0)
#endif

/* ── Direct-state-access helpers (GL 4.5, not in GLES3) ─────────────────*/
#ifndef glNamedFramebufferDrawBuffer
#  define glNamedFramebufferDrawBuffer(fb,buf) glDrawBuffer(buf)
#endif
#ifndef glNamedFramebufferReadBuffer
#  define glNamedFramebufferReadBuffer(fb,src) glReadBuffer(src)
#endif

/* ── Fragment output location binding (desktop GL 3.0) ───────────────────
 * glBindFragDataLocation is absent from GLES3; layout(location=N) in the
 * shader handles output slot assignment instead. */
#ifndef glBindFragDataLocation
#  define glBindFragDataLocation(prog,colNum,name) ((void)0)
#endif

/* ── Double-precision vertex attributes (GL 4.1, not in GLES3) ───────────
 * Fall back to the float variant; precision may differ but it compiles. */
#ifndef glVertexAttribLPointer
#  define glVertexAttribLPointer(idx,sz,type,stride,ptr) \
     glVertexAttribPointer(idx, sz, GL_FLOAT, GL_FALSE, stride, ptr)
#endif

/* ── Image copy (GLES3.2+ / GL_EXT_copy_image) ────────────────────────────
 * glCopyImageSubData is guarded by HAVE_OPENGL which we set; stub it out on
 * GLES3.0/3.1 where the function is not available. */
#ifndef glCopyImageSubData
#  define glCopyImageSubData(sn,st,sl,sx,sy,sz,dn,dt,dl,dx,dy,dz,w,h,d) \
     ((void)0)
#endif

/* ── Sample shading (GLES3.2 only) ───────────────────────────────────────*/
#ifndef glMinSampleShading
#  define glMinSampleShading(v) ((void)(v))
#endif

/* ── Enum values not defined in GLES3 headers ────────────────────────────*/
#ifndef GL_WRITE_ONLY
#  define GL_WRITE_ONLY 0x88B9
#endif
#ifndef GL_READ_ONLY
#  define GL_READ_ONLY  0x88B8
#endif
#ifndef GL_LINES_ADJACENCY
#  define GL_LINES_ADJACENCY 0x000A
#endif
#ifndef GL_LINE_STRIP_ADJACENCY
#  define GL_LINE_STRIP_ADJACENCY 0x000B
#endif
#ifndef GL_TRIANGLES_ADJACENCY
#  define GL_TRIANGLES_ADJACENCY 0x000C
#endif
#ifndef GL_TRIANGLE_STRIP_ADJACENCY
#  define GL_TRIANGLE_STRIP_ADJACENCY 0x000D
#endif
