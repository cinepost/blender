#ifndef __EEVEE_OCLLUSION_TRACE_H__
#define __EEVEE_OCLLUSION_TRACE_H__

#include <embree3/rtcore.h>

#include "GPU_texture.h"

#include "eevee_embree.h"
#include "eevee_private.h"

/* Maximum number of FBOs a texture can be attached to. */
#define GPU_TEX_MAX_FBO_ATTACHED 12

typedef enum eGPUTextureFormatFlag {
  GPU_FORMAT_DEPTH = (1 << 0),
  GPU_FORMAT_STENCIL = (1 << 1),
  GPU_FORMAT_INTEGER = (1 << 2),
  GPU_FORMAT_FLOAT = (1 << 3),

  GPU_FORMAT_1D = (1 << 10),
  GPU_FORMAT_2D = (1 << 11),
  GPU_FORMAT_3D = (1 << 12),
  GPU_FORMAT_CUBE = (1 << 13),
  GPU_FORMAT_ARRAY = (1 << 14),
} eGPUTextureFormatFlag;

/* GPUTexture */
struct GPUTexture {
  int w, h, d;        /* width/height/depth */
  int orig_w, orig_h; /* width/height (of source data), optional. */
  int number;         /* number for multitexture binding */
  int refcount;       /* reference count */
  GLenum target;      /* GL_TEXTURE_* */
  GLenum target_base; /* same as target, (but no multisample)
                       * use it for unbinding */
  GLuint bindcode;    /* opengl identifier for texture */

  eGPUTextureFormat format;
  eGPUTextureFormatFlag format_flag;

  int mipmaps;    /* number of mipmaps */
  int components; /* number of color/alpha channels */
  int samples;    /* number of samples for multisamples textures. 0 if not multisample target */

  int fb_attachment[GPU_TEX_MAX_FBO_ATTACHED];
  GPUFrameBuffer *fb[GPU_TEX_MAX_FBO_ATTACHED];
};

void EEVEE_rtao_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool cast_shadow);

void EEVEE_occlusion_rtao_texture_update(GPUTexture *tex, eGPUDataFormat data_format, const void *pixels);

void EEVEE_occlusion_rtao_read_gpu_buffers(EEVEE_Data *vedata);
void EEVEE_occlusion_rtao_read_gpu_buffers_fast(EEVEE_Data *vedata);

void PVZ_occlusion_trace_buffers_init(uint w, uint h);
void PVZ_occlusion_trace_compute_embree(void);
void PVZ_occlusion_trace_build_prim_rays_cpu(void);
void PVZ_occlusion_trace_build_prim_rays_gpu(void);
void PVZ_occlusion_trace_buffers_free(void);

#endif // __EEVEE_OCLLUSION_TRACE_H__