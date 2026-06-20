#include <glsm/glsm.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#define CHECK_GL(msg) { GLenum err; while((err = glGetError()) != GL_NO_ERROR) { __android_log_print(ANDROID_LOG_ERROR, "melonDS-GLES", "GL ERROR: 0x%04x at %s", err, msg); } }

#include "input.h"
#include "libretro_state.h"
#include "screenlayout.h"
#include "utils.h"

#include "Config.h"
#include "NDS.h"
#include "GPU.h"
#include "OpenGLSupport.h"
#include "shaders.h"

extern bool enable_opengl;
extern bool using_opengl;
extern bool refresh_opengl;
extern bool opengl_linear_filtering;

// M27 (GPU2D_OpenGL.cpp): join + GPU-order the deferred worker composite
// before the present samples the GL2D output textures.
namespace GPU2D { void M27SyncForConsume(); }

static bool initialized_glsm;
static GLuint shader[3];
static GLuint screen_framebuffer_texture;
static float screen_vertices[72];
static GLuint vao, vbo;

struct
{
   GLfloat uScreenSize[2];
   u32 u3DScale;
   u32 uFilterMode;
   GLfloat cursorPos[4];

} GL_ShaderConfig;
static GLuint ubo;

static bool setup_opengl(void)
{
   GPU::InitRenderer(true);
   GPU::SetRenderSettings(true, video_settings);

   if (!OpenGL::BuildShaderProgram(vertex_shader, fragment_shader, shader, "LibretroShader"))
      return false;

   glBindAttribLocation(shader[2], 0, "vPosition");
   glBindAttribLocation(shader[2], 1, "vTexcoord");

   if (!OpenGL::LinkShaderProgram(shader))
      return false;

   GLuint uni_id;

   uni_id = glGetUniformBlockIndex(shader[2], "uConfig");
   if (uni_id != GL_INVALID_INDEX)
      glUniformBlockBinding(shader[2], uni_id, 16);
   CHECK_GL("glUniformBlockBinding(uConfig)");

   glUseProgram(shader[2]);
   uni_id = glGetUniformLocation(shader[2], "ScreenTex");
   if ((GLint)uni_id >= 0)
      glUniform1i(uni_id, 0);
   CHECK_GL("glUniform1i(ScreenTex)");

   memset(&GL_ShaderConfig, 0, sizeof(GL_ShaderConfig));

   glGenBuffers(1, &ubo);
   glBindBuffer(GL_UNIFORM_BUFFER, ubo);
   glBufferData(GL_UNIFORM_BUFFER, sizeof(GL_ShaderConfig), &GL_ShaderConfig, GL_STATIC_DRAW);
   glBindBufferBase(GL_UNIFORM_BUFFER, 16, ubo);

   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(screen_vertices), NULL, GL_STATIC_DRAW);

   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao); CHECK_GL("glBindVertexArray");
   glEnableVertexAttribArray(0); // position
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*4, (void*)(0));
   glEnableVertexAttribArray(1); // texcoord
   glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*4, (void*)(2*4));

   glGenTextures(1, &screen_framebuffer_texture);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, screen_framebuffer_texture);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256*3 + 1, 192*2, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

   refresh_opengl = true;

   return true;
}

static void context_reset(void)
{
   if(using_opengl)
      GPU::DeInitRenderer();

   glsm_ctl(GLSM_CTL_STATE_CONTEXT_RESET, NULL);

   if (!glsm_ctl(GLSM_CTL_STATE_SETUP, NULL))
      return;

   glsm_ctl(GLSM_CTL_STATE_BIND, NULL);
   if (!setup_opengl())
   {
      glsm_ctl(GLSM_CTL_STATE_UNBIND, NULL);
      initialized_glsm = false;
      using_opengl = false;
      return;
   }

   glsm_ctl(GLSM_CTL_STATE_UNBIND, NULL);

   initialized_glsm = true;
   using_opengl = true;
}

static void context_destroy(void)
{
   glsm_ctl(GLSM_CTL_STATE_BIND, NULL);
   glDeleteTextures(1, &screen_framebuffer_texture);

   glDeleteVertexArrays(1, &vao);
   glDeleteBuffers(1, &vbo);

   OpenGL::DeleteShaderProgram(shader);
   glsm_ctl(GLSM_CTL_STATE_UNBIND, NULL);

   initialized_glsm = false;
}

static bool context_framebuffer_lock(void *data)
{
    return false;
}

bool initialize_opengl()
{
   glsm_ctx_params_t params = {0};

   // melonds wants an opengl 3.1 context, so glcore is required for mesa compatibility
   params.context_type     = RETRO_HW_CONTEXT_OPENGL_CORE;
   params.major            = 3;
   params.minor            = 1;
   params.context_reset    = context_reset;
   params.context_destroy  = context_destroy;
   params.environ_cb       = environ_cb;
   params.stencil          = false;
   params.framebuffer_lock = context_framebuffer_lock;

   if (!glsm_ctl(GLSM_CTL_STATE_CONTEXT_INIT, &params))
   {
      log_cb(RETRO_LOG_ERROR, "Could not setup opengl context, falling back to software rasterization.\n");
      return false;
   }

   return true;
}

void deinitialize_opengl_renderer(void)
{
   GPU::DeInitRenderer();
   GPU::InitRenderer(false);
}

void setup_opengl_frame_state(void)
{
   refresh_opengl = false;

   GPU::SetRenderSettings(true, video_settings);

   GL_ShaderConfig.uScreenSize[0] = (float)screen_layout_data.buffer_width;
   GL_ShaderConfig.uScreenSize[1] = (float)screen_layout_data.buffer_height;
   GL_ShaderConfig.u3DScale = (float)video_settings.GL_ScaleFactor;
   GL_ShaderConfig.cursorPos[0] = -1.0f;
   GL_ShaderConfig.cursorPos[1] = -1.0f;
   GL_ShaderConfig.cursorPos[2] = -1.0f;
   GL_ShaderConfig.cursorPos[3] = -1.0f;

   glBindBuffer(GL_UNIFORM_BUFFER, ubo);
   void* unibuf = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(GL_ShaderConfig), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
   if (unibuf) memcpy(unibuf, &GL_ShaderConfig, sizeof(GL_ShaderConfig));
   glUnmapBuffer(GL_UNIFORM_BUFFER); CHECK_GL("glUnmapBuffer");

   float screen_width = (float)screen_layout_data.screen_width;
   float screen_height = (float)screen_layout_data.screen_height;
   float screen_gap = (float)screen_layout_data.screen_gap;

   float top_screen_x = 0.0f;
   float top_screen_y = 0.0f;
   float top_screen_scale = 1.0f;

   float bottom_screen_x = 0.0f;
   float bottom_screen_y = 0.0f;
   float bottom_screen_scale = 1.0f;

   float primary_x = 0.0f;
   float primary_y = 0.0f;
   float primary_tex_v0_x = 0.0f;
   float primary_tex_v0_y = 0.0f;
   float primary_tex_v1_x = 0.0f;
   float primary_tex_v1_y = 0.0f;
   float primary_tex_v2_x = 0.0f;
   float primary_tex_v2_y = 0.0f;
   float primary_tex_v3_x = 0.0f;
   float primary_tex_v3_y = 0.0f;
   float primary_tex_v4_x = 0.0f;
   float primary_tex_v4_y = 0.0f;
   float primary_tex_v5_x = 0.0f;
   float primary_tex_v5_y = 0.0f;

   const float pixel_pad = 1.0f / (192 * 2 + 2);

   switch (screen_layout_data.displayed_layout)
   {
      case ScreenLayout::TopBottom:
         bottom_screen_y = screen_height + screen_gap;
         break;
      case ScreenLayout::BottomTop:
         top_screen_y = screen_height + screen_gap;
         break;
      case ScreenLayout::LeftRight:
         bottom_screen_x = screen_width;
         break;
      case ScreenLayout::RightLeft:
         top_screen_x = screen_width;
         break;
      case ScreenLayout::TopOnly:
         bottom_screen_y = screen_height; // Meh, let's just hide it
         break;
      case ScreenLayout::BottomOnly:
         top_screen_y = screen_height; // ditto
         break;
      case ScreenLayout::HybridTop:
         primary_x = screen_width * screen_layout_data.hybrid_ratio;
         primary_y = screen_height * screen_layout_data.hybrid_ratio;

         primary_tex_v0_x = 0.0f;
         primary_tex_v0_y = 0.0f;
         primary_tex_v1_x = 0.0f;
         primary_tex_v1_y = 0.5f - pixel_pad;
         primary_tex_v2_x = 1.0f;
         primary_tex_v2_y = 0.5f - pixel_pad;
         primary_tex_v3_x = 0.0f;
         primary_tex_v3_y = 0.0f;
         primary_tex_v4_x = 1.0f;
         primary_tex_v4_y = 0.0f;
         primary_tex_v5_x = 1.0f;
         primary_tex_v5_y = 0.5f - pixel_pad;

         break;
      case ScreenLayout::HybridBottom:
         primary_x = screen_width * screen_layout_data.hybrid_ratio;
         primary_y = screen_height * screen_layout_data.hybrid_ratio;

         primary_tex_v0_x = 0.0f;
         primary_tex_v0_y = 0.5f + pixel_pad;
         primary_tex_v1_x = 0.0f;
         primary_tex_v1_y = 1.0f;
         primary_tex_v2_x = 1.0f;
         primary_tex_v2_y = 1.0f;
         primary_tex_v3_x = 0.0f;
         primary_tex_v3_y = 0.5f + pixel_pad;
         primary_tex_v4_x = 1.0f;
         primary_tex_v4_y = 0.5f + pixel_pad;
         primary_tex_v5_x = 1.0f;
         primary_tex_v5_y = 01.0;

         break;
   }

   #define SETVERTEX(i, x, y, t_x, t_y) \
      screen_vertices[(4 * i) + 0] = x; \
      screen_vertices[(4 * i) + 1] = y; \
      screen_vertices[(4 * i) + 2] = t_x; \
      screen_vertices[(4 * i) + 3] = t_y;


   if (screen_layout_data.displayed_layout == ScreenLayout::HybridTop || screen_layout_data.displayed_layout == ScreenLayout::HybridBottom)
   {
      //Primary Screen
      SETVERTEX(0, 0.0f, 0.0f, primary_tex_v0_x, primary_tex_v0_y); // top left
      SETVERTEX(1, 0.0f, primary_y, primary_tex_v1_x, primary_tex_v1_y); // bottom left
      SETVERTEX(2, primary_x, primary_y, primary_tex_v2_x, primary_tex_v2_y); // bottom right
      SETVERTEX(3, 0.0f, 0.0f, primary_tex_v3_x, primary_tex_v3_y); // top left
      SETVERTEX(4, primary_x, 0.0f, primary_tex_v4_x, primary_tex_v4_y); // top right
      SETVERTEX(5, primary_x, primary_y, primary_tex_v5_x, primary_tex_v5_y); // bottom right

      //Top screen
      if(screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenTop && screen_layout_data.displayed_layout == ScreenLayout::HybridTop)
      {
         SETVERTEX(6, primary_x, 0.0f, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(7, primary_x, 0.0f + screen_height, 0.0f, 1.0f); // bottom left
         SETVERTEX(8, primary_x + screen_width, 0.0f + screen_height, 1.0f, 1.0f); // bottom right
         SETVERTEX(9, primary_x, 0.0f, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(10, primary_x + screen_width, 0.0f, 1.0f, 0.5f + pixel_pad); // top right
         SETVERTEX(11, primary_x + screen_width, 0.0f + screen_height, 1.0f, 1.0f); // bottom right
      }
      else if (screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenDuplicate
         || (screen_layout_data.displayed_layout == ScreenLayout::HybridBottom && screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenTop))
      {
         SETVERTEX(6, primary_x, 0.0f, 0.0f, 0.0f); // top left
         SETVERTEX(7, primary_x, 0.0f + screen_height, 0.0f, 0.5f - pixel_pad); // bottom left
         SETVERTEX(8, primary_x + screen_width, 0.0f + screen_height, 1.0f, 0.5f - pixel_pad); // bottom right
         SETVERTEX(9, primary_x, 0.0f, 0.0f, 0.0f); // top left
         SETVERTEX(10, primary_x + screen_width, 0.0f, 1.0f, 0.0f); // top right
         SETVERTEX(11, primary_x + screen_width, 0.0f + screen_height, 1.0f, 0.5f - pixel_pad); // bottom right
      }
      

      //Bottom Screen
      if(screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenBottom && screen_layout_data.displayed_layout == ScreenLayout::HybridTop)
      {
         SETVERTEX(6, primary_x, primary_y - screen_height, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(7, primary_x, primary_y, 0.0f, 1.0f); // bottom left
         SETVERTEX(8, primary_x + screen_width, primary_y, 1.0f, 1.0f); // bottom right
         SETVERTEX(9, primary_x, primary_y - screen_height, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(10, primary_x + screen_width, primary_y - screen_height, 1.0f, 0.5f + pixel_pad); // top right
         SETVERTEX(11, primary_x + screen_width, primary_y,  1.0f, 1.0f); // bottom right
      }
      else if(screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenBottom && screen_layout_data.displayed_layout == ScreenLayout::HybridBottom)
      {
         SETVERTEX(6, primary_x, primary_y - screen_height, 0.0f, 0.0f); // top left
         SETVERTEX(7, primary_x, primary_y, 0.0f, 0.5f - pixel_pad); // bottom left
         SETVERTEX(8, primary_x + screen_width, primary_y, 1.0f, 0.5f - pixel_pad); // bottom right
         SETVERTEX(9, primary_x, primary_y - screen_height, 0.0f, 0.0f); // top left
         SETVERTEX(10, primary_x + screen_width, primary_y - screen_height, 1.0f, 0.0f); // top right
         SETVERTEX(11, primary_x + screen_width, primary_y, 1.0f, 0.5f - pixel_pad); // bottom right
      }
      else if (screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenDuplicate)
      {
         SETVERTEX(12, primary_x, primary_y - screen_height, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(13, primary_x, primary_y, 0.0f, 1.0f); // bottom left
         SETVERTEX(14, primary_x + screen_width, primary_y, 1.0f, 1.0f); // bottom right
         SETVERTEX(15, primary_x, primary_y - screen_height, 0.0f, 0.5f + pixel_pad); // top left
         SETVERTEX(16, primary_x + screen_width, primary_y - screen_height, 1.0f, 0.5f + pixel_pad); // top right
         SETVERTEX(17, primary_x + screen_width, primary_y,  1.0f, 1.0f); // bottom right
      }
   }
   else
   {
      // top screen
      SETVERTEX(0, top_screen_x, top_screen_y, 0.0f, 0.0f); // top left
      SETVERTEX(1, top_screen_x, top_screen_y + screen_height * top_screen_scale, 0.0f, 0.5f - pixel_pad); // bottom left
      SETVERTEX(2, top_screen_x + screen_width * top_screen_scale, top_screen_y + screen_height * top_screen_scale, 1.0f, 0.5f - pixel_pad); // bottom right
      SETVERTEX(3, top_screen_x, top_screen_y, 0.0f, 0.0f); // top left
      SETVERTEX(4, top_screen_x + screen_width * top_screen_scale, top_screen_y, 1.0f, 0.0f); // top right
      SETVERTEX(5, top_screen_x + screen_width * top_screen_scale, top_screen_y + screen_height * top_screen_scale, 1.0f, 0.5f - pixel_pad); // bottom right

      // bottom screen
      SETVERTEX(6, bottom_screen_x, bottom_screen_y, 0.0f, 0.5f + pixel_pad); // top left
      SETVERTEX(7, bottom_screen_x, bottom_screen_y + screen_height * bottom_screen_scale, 0.0f, 1.0f); // bottom left
      SETVERTEX(8, bottom_screen_x + screen_width * bottom_screen_scale, bottom_screen_y + screen_height * bottom_screen_scale, 1.0f, 1.0f); // bottom right
      SETVERTEX(9, bottom_screen_x, bottom_screen_y, 0.0f, 0.5f + pixel_pad); // top left
      SETVERTEX(10, bottom_screen_x + screen_width * bottom_screen_scale, bottom_screen_y, 1.0f, 0.5f + pixel_pad); // top right
      SETVERTEX(11, bottom_screen_x + screen_width * bottom_screen_scale, bottom_screen_y + screen_height * bottom_screen_scale, 1.0f, 1.0f); // bottom right
   }

   // top screen


   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(screen_vertices), screen_vertices);
}

static void remap_gl2d_triangle(float* dst, int tri, bool bottom)
{
   for (int j = 0; j < 3; j++)
   {
      int idx = tri + j;
      float ty = screen_vertices[(4 * idx) + 3];

      dst[(4 * idx) + 0] = screen_vertices[(4 * idx) + 0];
      dst[(4 * idx) + 1] = screen_vertices[(4 * idx) + 1];
      dst[(4 * idx) + 2] = screen_vertices[(4 * idx) + 2];
      dst[(4 * idx) + 3] = bottom ? ((ty - 0.5f) * 2.0f) : (ty * 2.0f);
   }
}

static bool render_gl2d_output(void)
{
   // M27: if the GL2D composite was deferred to the frontend's render worker,
   // join it and order its GPU writes before we sample OutputTex below.
   GPU2D::M27SyncForConsume();

   float gl2d_vertices[72] = {};
   int vertex_count = screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenDuplicate ? 18 : 12;

   for (int tri = 0; tri < vertex_count; tri += 3)
   {
      float ty = 0.0f;
      for (int j = 0; j < 3; j++)
         ty += screen_vertices[(4 * (tri + j)) + 3];

      remap_gl2d_triangle(gl2d_vertices, tri, (ty / 3.0f) >= 0.5f);
   }

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(gl2d_vertices), gl2d_vertices);
   glBindVertexArray(vao); CHECK_GL("glBindVertexArray(GL2D)");

   GLint filter = opengl_linear_filtering ? GL_LINEAR : GL_NEAREST;
   for (int tri = 0; tri < vertex_count; tri += 3)
   {
      float ty = 0.0f;
      for (int j = 0; j < 3; j++)
         ty += screen_vertices[(4 * (tri + j)) + 3];

      bool bottom = (ty / 3.0f) >= 0.5f;
      glActiveTexture(GL_TEXTURE0);
      if (!GPU::Bind2DOutputTextureForScreen(bottom ? 1 : 0))
         return false;

      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
      glDrawArrays(GL_TRIANGLES, tri, 3); CHECK_GL("glDrawArrays(GL2D)");
   }

   return true;
}

static void prepare_default_present_state(void)
{
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   glDisable(GL_BLEND);
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_CULL_FACE);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDepthMask(GL_TRUE);
   glStencilMask(0xFF);

#if defined(HAVE_OPENGLES3)
   glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   const GLenum back = GL_BACK;
   glDrawBuffers(1, &back);
#elif defined(HAVE_OPENGL)
   glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glDrawBuffer(GL_BACK);
#endif
}

void render_opengl_frame(bool sw)
{
   bool gl2d_active = GPU::GL2DActive;
   if (!gl2d_active)
      glsm_ctl(GLSM_CTL_STATE_BIND, NULL);

   int frontbuf = GPU::FrontBuffer;
   bool virtual_cursor = cursor_enabled(&input_state);

   glBindFramebuffer(GL_FRAMEBUFFER, glsm_get_current_framebuffer()); CHECK_GL("glBindFramebuffer");
   prepare_default_present_state(); CHECK_GL("prepare_default_present_state");

   static unsigned last_buffer_width = 0;
   static unsigned last_buffer_height = 0;
   static unsigned last_screen_width = 0;
   static unsigned last_screen_height = 0;
   static unsigned last_screen_gap = 0;
   static unsigned last_hybrid_ratio = 0;
   static ScreenLayout last_layout = ScreenLayout::TopBottom;
   static SmallScreenLayout last_small_screen = SmallScreenLayout::SmallScreenTop;

   bool layout_dirty =
      refresh_opengl ||
      last_buffer_width != screen_layout_data.buffer_width ||
      last_buffer_height != screen_layout_data.buffer_height ||
      last_screen_width != screen_layout_data.screen_width ||
      last_screen_height != screen_layout_data.screen_height ||
      last_screen_gap != screen_layout_data.screen_gap ||
      last_hybrid_ratio != screen_layout_data.hybrid_ratio ||
      last_layout != screen_layout_data.displayed_layout ||
      last_small_screen != screen_layout_data.hybrid_small_screen;

   if(layout_dirty)
   {
      setup_opengl_frame_state();
      last_buffer_width = screen_layout_data.buffer_width;
      last_buffer_height = screen_layout_data.buffer_height;
      last_screen_width = screen_layout_data.screen_width;
      last_screen_height = screen_layout_data.screen_height;
      last_screen_gap = screen_layout_data.screen_gap;
      last_hybrid_ratio = screen_layout_data.hybrid_ratio;
      last_layout = screen_layout_data.displayed_layout;
      last_small_screen = screen_layout_data.hybrid_small_screen;
   }

   if(virtual_cursor)
   {
      GL_ShaderConfig.cursorPos[0] = (((float)(input_state.touch_x) - (float)(CURSOR_SIZE)) / ((float)VIDEO_HEIGHT * 1.35));
      GL_ShaderConfig.cursorPos[1] = (((float)(input_state.touch_y) - (float)(CURSOR_SIZE)) / ((float)VIDEO_WIDTH * 1.5)) + 0.5f;
      GL_ShaderConfig.cursorPos[2] = (((float)(input_state.touch_x) + (float)(CURSOR_SIZE)) / ((float)VIDEO_HEIGHT * 1.35));
      GL_ShaderConfig.cursorPos[3] = (((float)(input_state.touch_y) + (float)(CURSOR_SIZE)) / ((float)VIDEO_WIDTH * 1.5)) + 0.5f;

      glBindBuffer(GL_UNIFORM_BUFFER, ubo);
      void* unibuf = glMapBufferRange(GL_UNIFORM_BUFFER, 0, sizeof(GL_ShaderConfig), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
      if (unibuf) memcpy(unibuf, &GL_ShaderConfig, sizeof(GL_ShaderConfig));
      glUnmapBuffer(GL_UNIFORM_BUFFER); CHECK_GL("glUnmapBuffer");
   }

   OpenGL::UseShaderProgram(shader); CHECK_GL("UseShaderProgram");

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_STENCIL_TEST);
   glDisable(GL_BLEND);

   glViewport(0, 0, screen_layout_data.buffer_width, screen_layout_data.buffer_height); CHECK_GL("glViewport");
   glClearColor(0, 0, 0, 1);
   glClear(GL_COLOR_BUFFER_BIT); CHECK_GL("glClear(default framebuffer)");

   glActiveTexture(GL_TEXTURE0);

   if (gl2d_active)
   {
      if (!render_gl2d_output())
      {
         if (!gl2d_active)
            glsm_ctl(GLSM_CTL_STATE_UNBIND, NULL);
         return;
      }
   }
   else if (sw)
   {
      glBindTexture(GL_TEXTURE_2D, screen_framebuffer_texture);

      if (GPU::Framebuffer[frontbuf][0] && GPU::Framebuffer[frontbuf][1])
      {
         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 192, GL_RGBA,
                        GL_UNSIGNED_BYTE, GPU::Framebuffer[frontbuf][0]);
         glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 192, 256, 192, GL_RGBA,
                        GL_UNSIGNED_BYTE, GPU::Framebuffer[frontbuf][1]);
      }
   }
   else
   {
      GPU::CurGLCompositor->BindOutputTexture(frontbuf); CHECK_GL("BindOutputTexture");
   }

   if (!gl2d_active)
   {
      GLint filter = opengl_linear_filtering ? GL_LINEAR : GL_NEAREST;
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);

      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBindVertexArray(vao); CHECK_GL("glBindVertexArray");
      glDrawArrays(GL_TRIANGLES, 0, screen_layout_data.hybrid_small_screen == SmallScreenLayout::SmallScreenDuplicate ? 18 : 12); CHECK_GL("glDrawArrays");
   }

   glFlush();

   if (!gl2d_active)
      glsm_ctl(GLSM_CTL_STATE_UNBIND, NULL);

   video_cb(RETRO_HW_FRAME_BUFFER_VALID, screen_layout_data.buffer_width, screen_layout_data.buffer_height, 0);
}
