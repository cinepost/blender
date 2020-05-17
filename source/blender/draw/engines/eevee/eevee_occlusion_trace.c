/*
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * Copyright 2016, Blender Foundation.
 */

/** \file
 * \ingroup draw_engine
 *
 * Implementation of the screen space Ground Truth Ambient Occlusion.
 */

#include "DRW_render.h"
#include "DRW_engine.h"

#include "BLI_string_utils.h"

#include "DEG_depsgraph_query.h"

#include "BKE_global.h" /* for G.debug_value */

#include "eevee_private.h"

#include "draw_manager.h"

#include "GPU_buffers.h"
#include "GPU_extensions.h"
#include "GPU_platform.h"
#include "GPU_state.h"
#include "GPU_texture.h"

#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "pthread.h"
#include "eevee_embree.h"
#include "eevee_occlusion_trace.h"

#define NUM_THREADS 8
#define RAYS_STREAM_SIZE 2560
#define EMBREE_TRACE_BIAS 0.001
#define TRACE_BIAS 0.005

static struct EEVEE_EmbreeDenoiseUniformBufferData {
  float stepwidth;
  float c_phi;
  float n_phi;
  float p_phi;
} denoise_uniform_buffer_data = {.stepwidth = 1.0, .c_phi = 0.5, .n_phi = 0.5, .p_phi = 0.5};

static struct {
  /* Ground Truth Ambient Occlusion */
  struct GPUShader *gtao_sh;
  struct GPUShader *gtao_debug_sh;

  struct GPUShader *rtao_pos_norm_sh;
  struct GPUShader *rtao_denoise_sh;

  struct GPUUniformBuffer *denoise_ubo;
  struct DRWShadingGroup  *shgrp;

  Object *camera;
  bool use_gpu_buff;
  float ao_dist;
  float gpu_bias;
  bool denoise;
  bool embree_mode;
} e_data = {NULL, .denoise = false, .gpu_bias = 0.005, .embree_mode = true}; /* Engine data */

static struct {
  uint w, h; /* cpu buffer width, height*/
  struct RTCRay *rays;
  struct RTCRayHit *rayhits;
  unsigned char *hits; /* embree occlusion hits buffer */ // TODO: try pack 4or8 hits in one byte
  float *norm;   /* world normal */
  float *pos;    /* world pos */
} rtao_cpu_buff = {.hits = NULL, .w=0, .h=0}; /* CPU ao data */

static struct {
  //float *kernel;
  //int   *offset;

  unsigned int iterations;
  float c_phi, n_phi, p_phi;

} rtao_deonise_data = {NULL};

static struct RTCIntersectContext embree_context;

extern struct EeveeEmbreeData evem_data;

static struct EeveeEmbreeRaysBuffer prim_rays_buff = {.rays=NULL, .rays16=NULL, .rays8=NULL, .rays4=NULL, .w=0, .h=0};

extern char datatoc_gpu_shader_3D_vert_glsl[];
extern char datatoc_ambient_occlusion_lib_glsl[];
extern char datatoc_ambient_occlusion_embree_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];
extern char datatoc_common_uniforms_lib_glsl[];
extern char datatoc_common_hair_lib_glsl[];
extern char datatoc_bsdf_common_lib_glsl[];
extern char datatoc_effect_gtao_embree_frag_glsl[];
extern char datatoc_effect_gtao_embree_vert_glsl[];
extern char datatoc_effect_gtao_embree_denoise_frag_glsl[];

static void eevee_create_shader_occlusion_trace(void)
{
  char *frag_str = BLI_string_joinN(datatoc_common_view_lib_glsl,
                                    datatoc_common_uniforms_lib_glsl,
                                    datatoc_bsdf_common_lib_glsl,
                                    datatoc_ambient_occlusion_lib_glsl,
                                    datatoc_ambient_occlusion_embree_lib_glsl,
                                    datatoc_effect_gtao_embree_frag_glsl);

  char *vert_str = BLI_string_joinN(datatoc_common_view_lib_glsl, 
                                    datatoc_common_hair_lib_glsl, 
                                    datatoc_effect_gtao_embree_vert_glsl);

  char *denoise_frag_str = BLI_string_joinN(
                                    datatoc_effect_gtao_embree_denoise_frag_glsl);

  e_data.gtao_sh = DRW_shader_create_fullscreen(frag_str, NULL);
  
  //e_data.rtao_pos_norm_sh = DRW_shader_create_fullscreen(frag_str, "#define AO_TRACE_POS\n");
  e_data.rtao_pos_norm_sh = GPU_shader_create(vert_str, frag_str, NULL, NULL, "#define AO_TRACE_POS\n", "rtao_nd_shader");//__func__);

  e_data.gtao_debug_sh = DRW_shader_create_fullscreen(frag_str, "#define DEBUG_AO\n");
  e_data.rtao_denoise_sh = DRW_shader_create_fullscreen(denoise_frag_str, NULL);

  MEM_freeN(frag_str);
  MEM_freeN(vert_str);
  MEM_freeN(denoise_frag_str);
}

int EEVEE_occlusion_trace_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata, Object *camera)
{
  printf("%s\n", "EEVEE_occlusion_trace_init");
  omp_set_num_threads(8);
  //printf("OpenMP num threads: %d\n", omp_get_num_threads());
  //printf("OpenMP max threads: %d\n", omp_get_max_threads());
  e_data.camera = camera;

  EEVEE_CommonUniformBuffer *common_data = &sldata->common_data;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;
  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  e_data.denoise = (scene_eval->eevee.flag & SCE_EEVEE_RTAO_DENOISE) ? true : false;

  e_data.embree_mode = false;
  if (scene_eval->eevee.flag & SCE_EEVEE_GTAO_ENABLED) {
    e_data.embree_mode = true;

    const float *viewport_size = DRW_viewport_size_get();
    const int fs_size[2] = {(int)viewport_size[0], (int)viewport_size[1]};

    /* UI data fetch */
    rtao_deonise_data.iterations = pvz_max(1, (uint)scene_eval->eevee.rtao_denoise_iterations);
    rtao_deonise_data.c_phi = scene_eval->eevee.rtao_denoise_c_phi;
    rtao_deonise_data.n_phi = scene_eval->eevee.rtao_denoise_n_phi;
    rtao_deonise_data.p_phi = scene_eval->eevee.rtao_denoise_p_phi;

    e_data.gpu_bias = scene_eval->eevee.rtao_gpubuff_bias;

    /* Some usefull info*/
    e_data.use_gpu_buff = false;
    if(scene_eval->eevee.flag & SCE_EEVEE_RTAO_GPUBUFF)
      e_data.use_gpu_buff = true;

    /* CPU buffers*/
    PVZ_occlusion_trace_buffers_init((uint)viewport_size[0], (uint)viewport_size[1]);

    /* Denoising ubo data and buffer */
    if (!e_data.denoise_ubo) {
      e_data.denoise_ubo = DRW_uniformbuffer_create(sizeof(struct EEVEE_EmbreeDenoiseUniformBufferData), &denoise_uniform_buffer_data);
    }
    //if (e_data.denoise && !rtao_deonise_data.kernel) {
    //  for (uint i = 0; i < 25; i++) {
    //    rtao_deonise_data.kernel[i]
    //  }
    //}

    /* Shaders */
    if (!e_data.gtao_sh) {
      eevee_create_shader_occlusion_trace();
    }

    /* Common ubo data */
    common_data->ao_embree = 0.0f;
    common_data->ao_dist = scene_eval->eevee.gtao_distance;
    e_data.ao_dist = common_data->ao_dist / 2.0;
    common_data->ao_factor = scene_eval->eevee.gtao_factor;
    common_data->ao_quality = 1.0f - scene_eval->eevee.gtao_quality;

    common_data->ao_settings = 1.0f; /* USE_AO */
    if (scene_eval->eevee.flag & SCE_EEVEE_GTAO_BENT_NORMALS) {
      common_data->ao_settings += 2.0f; /* USE_BENT_NORMAL */
    }
    if (scene_eval->eevee.flag & SCE_EEVEE_GTAO_BOUNCE) {
      common_data->ao_settings += 4.0f; /* USE_DENOISE */
    }

    common_data->ao_bounce_fac = (scene_eval->eevee.flag & SCE_EEVEE_GTAO_BOUNCE) ? 1.0f : 0.0f;

    effects->rtao_embree_tx_1 = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_R8, &draw_engine_eevee_type);
    effects->rtao_embree_tx_2 = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_R8, &draw_engine_eevee_type);

    effects->rtao_nrm = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_RGBA16F, &draw_engine_eevee_type);
    effects->rtao_pos = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_RGBA32F, &draw_engine_eevee_type);
    GPU_framebuffer_ensure_config(&fbl->rtao_pos_norm_fb, {
      GPU_ATTACHMENT_TEXTURE(dtxl->depth),
      GPU_ATTACHMENT_TEXTURE(effects->rtao_nrm), 
      GPU_ATTACHMENT_TEXTURE(effects->rtao_pos)
    });

    if (e_data.denoise) {
      GPU_framebuffer_ensure_config(&fbl->rtao_denoise_fb_1, {
        GPU_ATTACHMENT_NONE,
        GPU_ATTACHMENT_TEXTURE(effects->rtao_embree_tx_2), 
      });
      GPU_framebuffer_ensure_config(&fbl->rtao_denoise_fb_2, {
        GPU_ATTACHMENT_NONE,
        GPU_ATTACHMENT_TEXTURE(effects->rtao_embree_tx_1), 
      });
    }

    effects->gtao_horizons_debug = NULL;

    return EFFECT_GTAO | EFFECT_NORMAL_BUFFER | EFFECT_GTAO_TRACE;
  }

  /* Cleanup */
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_pos_norm_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_denoise_fb_1);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_denoise_fb_2);
  common_data->ao_settings = 0.0f;

  return 0;
}

void EEVEE_occlusion_trace_output_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata, uint tot_samples)
{
  printf("%s\n", "EEVEE_occlusion_trace_output_init");
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_TextureList *txl = vedata->txl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_EffectsInfo *effects = stl->effects;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  if (scene_eval->eevee.flag & SCE_EEVEE_GTAO_ENABLED) {
    const eGPUTextureFormat texture_format = (tot_samples > 128) ? GPU_R32F : GPU_R16F;

    DefaultTextureList *dtxl = DRW_viewport_texture_list_get();
    float clear[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Should be enough precision for many samples. */
    DRW_texture_ensure_fullscreen_2d(&txl->ao_accum, texture_format, 0);

    GPU_framebuffer_ensure_config(&fbl->ao_accum_fb,
                                  {GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(txl->ao_accum)});

    /* Clear texture. */
    if (DRW_state_is_image_render() || effects->taa_current_sample == 1) {
      GPU_framebuffer_bind(fbl->ao_accum_fb);
      GPU_framebuffer_clear_color(fbl->ao_accum_fb, clear);
    }

    /* Accumulation pass */
    DRWState state = DRW_STATE_WRITE_COLOR | DRW_STATE_BLEND_ADD;
    DRW_PASS_CREATE(psl->ao_accum_ps, state);
    DRWShadingGroup *grp = DRW_shgroup_create(e_data.gtao_debug_sh, psl->ao_accum_ps);
    DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_texture_ref(grp, "maxzBuffer", &txl->maxzbuffer);
    DRW_shgroup_uniform_texture_ref(grp, "depthBuffer", &dtxl->depth);
    DRW_shgroup_uniform_texture_ref(grp, "normalBuffer", &effects->ssr_normal_input);
    DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeBuffer", &effects->rtao_embree_tx_final);
    DRW_shgroup_uniform_block(grp, "common_block", sldata->common_ubo);
    DRW_shgroup_uniform_block(
        grp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));
    DRW_shgroup_call(grp, DRW_cache_fullscreen_quad_get(), NULL);
  }
  else {
    /* Cleanup to release memory */
    DRW_TEXTURE_FREE_SAFE(txl->ao_accum);
    GPU_FRAMEBUFFER_FREE_SAFE(fbl->ao_accum_fb);
  }
}

void EEVEE_occlusion_trace_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata)
{
  printf("%s\n", "EEVEE_occlusion_trace_cache_init");
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_TextureList *txl = vedata->txl;
  EEVEE_EffectsInfo *effects = stl->effects;
  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

  struct GPUBatch *quad = DRW_cache_fullscreen_quad_get();

  if ((effects->enabled_effects & EFFECT_GTAO) != 0) {
    /**  Occlusion algorithm overview
     *
     *  We separate the computation into 2 steps.
     *
     * - First we scan the neighborhood pixels to find the maximum horizon angle.
     *   We save this angle in a RG8 array texture.
     *
     * - Then we use this angle to compute occlusion with the shading normal at
     *   the shading stage. This let us do correct shadowing for each diffuse / specular
     *   lobe present in the shader using the correct normal.
     */

    EEVEE_CommonUniformBuffer *common_data = &sldata->common_data;
    common_data->ao_embree = 1.0f;

    float sample_num = 0.0f;

    /* This pass is used to calculate ao rays positions on cpu */
    DRW_PASS_CREATE(psl->ao_trace_pos, DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_EQUAL | DRW_STATE_CLIP_PLANES | DRW_STATE_CULL_BACK);
    e_data.shgrp = DRW_shgroup_create(e_data.rtao_pos_norm_sh, psl->ao_trace_pos);
    DRW_shgroup_uniform_texture(e_data.shgrp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_block(e_data.shgrp, "common_block", sldata->common_ubo);
    DRW_shgroup_uniform_float(e_data.shgrp, "sampleNum", &sample_num, 1);
    //DRW_shgroup_uniform_texture_ref(e_data.shgrp, "depthBuffer", &effects->ao_src_depth);
    //DRW_shgroup_uniform_texture_ref(e_data.shgrp, "normalBuffer", &effects->ssr_normal_input); // we should calc world normals here on GPU it's faster
    DRW_shgroup_uniform_block(
        e_data.shgrp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));

    /* Desnoising passese and debug */
    DRWShadingGroup *grp = NULL;
    if (e_data.denoise) {
      /* two passes needed for several denoising iterations */
      DRW_PASS_CREATE(psl->ao_embree_denoise_pass1, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.rtao_denoise_sh, psl->ao_embree_denoise_pass1);
      DRW_shgroup_uniform_texture_ref(grp, "wposBuffer",  &effects->rtao_pos);
      DRW_shgroup_uniform_texture_ref(grp, "wnormBuffer", &effects->rtao_nrm);
      DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeRawBuffer", &effects->rtao_embree_tx_1);
      DRW_shgroup_uniform_float(grp, "sampleNum", &sample_num, 1);
      //DRW_shgroup_uniform_ivec2(grp, "offset", &, 50);
      //DRW_shgroup_uniform_float(grp, "kernel", &, 25);
      DRW_shgroup_uniform_block(grp, "denoise_block", e_data.denoise_ubo);
      DRW_shgroup_call(grp, quad, NULL);

      DRW_PASS_CREATE(psl->ao_embree_denoise_pass2, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.rtao_denoise_sh, psl->ao_embree_denoise_pass2);
      DRW_shgroup_uniform_texture_ref(grp, "wposBuffer",  &effects->rtao_pos);
      DRW_shgroup_uniform_texture_ref(grp, "wnormBuffer", &effects->rtao_nrm);
      DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeRawBuffer", &effects->rtao_embree_tx_2);
      DRW_shgroup_uniform_float(grp, "sampleNum", &sample_num, 1);
      //DRW_shgroup_uniform_ivec2(grp, "offset", &, 50);
      //DRW_shgroup_uniform_float(grp, "kernel", &, 25);
      DRW_shgroup_uniform_block(grp, "denoise_block", e_data.denoise_ubo);
      DRW_shgroup_call(grp, quad, NULL);

    }

    if (G.debug_value == 6) {
      DRW_PASS_CREATE(psl->ao_embree_debug, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.gtao_debug_sh, psl->ao_embree_debug);
      DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
      DRW_shgroup_uniform_texture_ref(grp, "maxzBuffer", &txl->maxzbuffer);
      DRW_shgroup_uniform_texture_ref(grp, "depthBuffer", &dtxl->depth);
      DRW_shgroup_uniform_texture_ref(grp, "normalBuffer", &effects->ssr_normal_input);
      DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeBuffer", &effects->rtao_embree_tx_final);
      DRW_shgroup_uniform_block(grp, "common_block", sldata->common_ubo);
      DRW_shgroup_uniform_block(
          grp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));
      DRW_shgroup_call(grp, quad, NULL);
    }
  }
}

void EEVEE_rtao_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool *cast_shadow) {
  if(!e_data.shgrp || !e_data.embree_mode) return;
  printf("%s\n", "EEVEE_rtao_cache_populate");
  EEVEE_PassList *psl = vedata->psl;

  struct GPUBatch *geom = NULL;
  geom = DRW_cache_object_surface_get(ob);
  DRW_shgroup_call(e_data.shgrp, geom, ob);
}

/*
 * Builds buffers that are necessary for our occlusion ray tracing
 */
void PVZ_occlusion_trace_buffers_init(uint w, uint h) {
  /* buffers exist and no size changes required */
  if((rtao_cpu_buff.hits != NULL) && rtao_cpu_buff.w == w && rtao_cpu_buff.h == h) {
    printf("%s\n", "reuse cpu buff");
    return;
  }

  /* viewport size changed */
  if(rtao_cpu_buff.hits != NULL) {
     PVZ_occlusion_trace_buffers_free();
  }  

  rtao_cpu_buff.w = w;
  rtao_cpu_buff.h = h;
  
  /* (re)create buffers  */
  rtao_cpu_buff.hits = (unsigned char *)malloc(sizeof(unsigned char) * w * h);
  rtao_cpu_buff.norm = malloc(sizeof(float) * w * h * 3);
  rtao_cpu_buff.pos = malloc(sizeof(float) * w * h * 3);
  rtao_cpu_buff.rays = (struct RTCRay *)malloc(sizeof(struct RTCRay) * w * h);
  rtao_cpu_buff.rayhits = (struct RTCRayHit *)malloc(sizeof(struct RTCRayHit) * w * h);
}

void PVZ_occlusion_trace_buffers_free(void) {
  rtao_cpu_buff.w = 0;
  rtao_cpu_buff.h = 0;
  if(rtao_cpu_buff.hits) free(rtao_cpu_buff.hits);
  if(rtao_cpu_buff.norm) free(rtao_cpu_buff.norm);
  if(rtao_cpu_buff.pos) free(rtao_cpu_buff.pos);
  if(rtao_cpu_buff.rays) free(rtao_cpu_buff.rays);
  if(rtao_cpu_buff.rayhits) free(rtao_cpu_buff.rayhits);
}

/* primary rays buffer */
void PVZ_occlusion_trace_build_prim_rays_buffer(uint w, uint h) {
  if (prim_rays_buff.w == w && prim_rays_buff.h == h)return; // buffer size not changed

  EVEM_rays_buffer_free(&prim_rays_buff); // free previous buffer

  prim_rays_buff.rays = (struct RTCRay *)malloc(sizeof(struct RTCRay) * w * h);

  prim_rays_buff.w = w;
  prim_rays_buff.h = h;
}

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")

void PVZ_occlusion_trace_build_prim_rays_cpu(void) {
  uint width = rtao_cpu_buff.w, height = rtao_cpu_buff.h;
  float scale = 0.5;
  float image_aspect_ratio = width / (float)height;

  EVEM_Matrix44f m;
  float cam_pos[3] = {0, 0, 0};

  if (e_data.camera) {
    cam_pos[0] = e_data.camera->loc[0];
    cam_pos[1] = e_data.camera->loc[1];
    cam_pos[2] = e_data.camera->loc[2];
    memcpy(&m[0][0], &e_data.camera->obmat[0][0], sizeof m);
  } else {
    const DRWContextState *draw_ctx = DRW_context_state_get();
    RegionView3D *rv3d = draw_ctx->rv3d;
    memcpy(&m[0][0], &rv3d->viewmat[0][0], sizeof m);
    cam_pos[0] = m[0][3]; 
    cam_pos[1] = m[1][3]; 
    cam_pos[2] = m[2][3]; 
  }

  uint ix, iy; // pixel pos
  EVEM_Vec3f v; // pixel pos in screen space
  EVEM_Vec3f dir; // ray direction

  struct RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  struct RTCRayHit *rayhits = rtao_cpu_buff.rayhits;
  #pragma omp parallel for num_threads(8) private(ix, iy, v, dir)
  for( int i=0; i < rtao_cpu_buff.w * rtao_cpu_buff.h; i++){
    rayhits[i].ray.id = i;
    ix = i - ((i / rtao_cpu_buff.w) * rtao_cpu_buff.w);
    iy = rtao_cpu_buff.h - i / rtao_cpu_buff.w;

    v[0] = (2 * (ix + 0.5) / (float)width - 1) * scale; 
    v[1] = (1 - 2 * (iy + 0.5) / (float)height) * scale * 1 / image_aspect_ratio;
    v[2] = -1.0;

    // matrix mul
    dir[0] = v[0] * m[0][0] + v[1] * m[1][0] + v[2] * m[2][0];
    dir[1] = v[0] * m[0][1] + v[1] * m[1][1] + v[2] * m[2][1]; 
    dir[2] = v[0] * m[0][2] + v[1] * m[1][2] + v[2] * m[2][2];
    
    // normalize dir
    float l = sqrt( dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] );
    dir[0] /= l; dir[1] /= l; dir[2] /= l; 

    rayhits[i].ray.mask = 0xFFFFFFFF;
    rayhits[i].ray.org_x = cam_pos[0];
    rayhits[i].ray.org_y = cam_pos[1];
    rayhits[i].ray.org_z = cam_pos[2];
    rayhits[i].ray.tnear = 0.0;

    rayhits[i].ray.dir_x = dir[0];
    rayhits[i].ray.dir_y = dir[1];
    rayhits[i].ray.dir_z = dir[2];
    rayhits[i].ray.tfar = 100000.0;
    rayhits[i].ray.time = 0.0;
  }

  struct RTCRay *rays = rtao_cpu_buff.rays;
  #pragma omp parallel for num_threads(8)// private(ray, hit)
  for(uint i=0; i < ((rtao_cpu_buff.w * rtao_cpu_buff.h)/RAYS_STREAM_SIZE); i++) {
    rtcIntersect1M(evem_data.scene, &context, &rayhits[i*RAYS_STREAM_SIZE], RAYS_STREAM_SIZE, sizeof(struct RTCRayHit));
    
    struct RTCRay *ray;
    struct RTCHit *hit;
    uint iix;
    uint iiy;
    uint ii;
    float n_dot_i;
    float nx, ny, nz;
    //#pragma omp simd private(ray, hit)
    for(int j=0; j<RAYS_STREAM_SIZE; j++){
      ii = j+i*RAYS_STREAM_SIZE;
      iix = ii - ((ii / rtao_cpu_buff.w) * rtao_cpu_buff.w);
      iiy = rtao_cpu_buff.h - ii / rtao_cpu_buff.w;

      ray = &rayhits[ii].ray;
      hit = &rayhits[ii].hit;

      rays[ii].id = ii;
      rays[ii].mask = 0xFFFFFFFF;
      rays[ii].tnear = 0.0;
      rays[ii].tfar = 10000.0;//e_data.ao_dist;
      rays[ii].time = 0.0;

      // face forward normal
      n_dot_i = ray->dir_x * hit->Ng_x + ray->dir_y * hit->Ng_y + ray->dir_z * hit->Ng_z; 

      nx = hit->Ng_x;
      ny = hit->Ng_y;
      nz = hit->Ng_z;

      if (n_dot_i > 0.0) {
        nx *= -1.0;
        ny *= -1.0;
        nz *= -1.0;
      }

      nx = 1.0;
      ny = 0.0;
      nz = 0.0;

      rays[ii].dir_x = nx;
      rays[ii].dir_y = ny;
      rays[ii].dir_z = nz;
      
      rays[ii].org_x = ray->org_x + ray->dir_x * ray->tfar + nx * EMBREE_TRACE_BIAS;
      rays[ii].org_y = ray->org_y + ray->dir_y * ray->tfar + ny * EMBREE_TRACE_BIAS;
      rays[ii].org_z = ray->org_z + ray->dir_z * ray->tfar + nz * EMBREE_TRACE_BIAS;
    }
  }
}

void PVZ_occlusion_trace_build_prim_rays_gpu(void) {
  struct RTCRay *rays = rtao_cpu_buff.rays;
  #pragma omp parallel for num_threads(8)
  for( uint i=0; i < rtao_cpu_buff.w * rtao_cpu_buff.h; i++){
    rays[i].id = i; // we need this as Embree might rearrange rays for better performance

    rays[i].mask = 0xFFFFFFFF;
    rays[i].org_x = rtao_cpu_buff.pos[i*3];
    rays[i].org_y = rtao_cpu_buff.pos[i*3+1];
    rays[i].org_z = rtao_cpu_buff.pos[i*3+2];;
    rays[i].tnear = pvz_max(0.00001f, e_data.gpu_bias);

    rays[i].dir_x = rtao_cpu_buff.norm[i*3];
    rays[i].dir_y = rtao_cpu_buff.norm[i*3+1];
    rays[i].dir_z = rtao_cpu_buff.norm[i*3+2];
    rays[i].tfar = e_data.ao_dist;
    rays[i].time = 0.0f;
  }
}

//EmbreeTraceRayStreamWorkerData *ao_thread_data = NULL;

/* calc rays stream in a thread */
void embree_trace_ray_stream_worker(void *arg) {
  //EmbreeTraceRayStreamWorkerData *args = (EmbreeTraceRayStreamWorkerData *)arg; 
  EmbreeTraceRayStreamWorkerData args = *(EmbreeTraceRayStreamWorkerData *)arg;      /* data received by thread */
  printf("worker - %u, job - %u \n", args.worker_id, args.job_id);
  rtcOccluded1M(args.scene, args.context, args.rays, args.stream_size, sizeof(struct RTCRay));
  //pthread_exit(NULL);
}

void PVZ_occlusion_trace_compute_embree(void) {
  printf("%s\n", "embree trace occlusion");
  struct timeval t_start, t_end;
  double elapsed_time;

  // start timer
  gettimeofday(&t_start, NULL);

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);


  struct RTCIntersectContext ctx;

  rtcInitIntersectContext(&ctx);
  ctx.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

  struct RTCRay *rays = rtao_cpu_buff.rays;

  uint num_pixels = rtao_cpu_buff.w * rtao_cpu_buff.h;

  RTCScene scene = evem_data.scene;
  #pragma omp parallel for num_threads(8) shared(scene, rays, ctx, rtao_cpu_buff)
  for(uint i=0; i < num_pixels; i+=RAYS_STREAM_SIZE) {
    rtcOccluded1M(scene, &ctx, &rays[i], RAYS_STREAM_SIZE, sizeof(struct RTCRay));
    
    struct RTCRay *ray = NULL;
    
    #pragma omp simd //private(i) //shared(rays, rtao_cpu_buff)
    for(uint j=0; j < RAYS_STREAM_SIZE; j++){
      ray = &rays[i + j];
      rtao_cpu_buff.hits[ray->id] = (ray->tfar >= 0.0f) ? 255 : 0;
    }
  }
  
  // stop timer
  gettimeofday(&t_end, NULL);
  elapsed_time = t_end.tv_sec + t_end.tv_usec / 1e6 - t_start.tv_sec - t_start.tv_usec / 1e6; // in seconds

  //free(ao_thread_data);
  printf("test trace occlusion done in %f seconds with %u rays \n", elapsed_time, rtao_cpu_buff.w*rtao_cpu_buff.h);
}

#pragma GCC pop_options

void PVZ_read_gpu_buffers(EEVEE_Data *vedata) {
  struct timeval t_start, t_end;
  double elapsed_time;

  // start timer
  gettimeofday(&t_start, NULL);

  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_EffectsInfo *effects = stl->effects;

  GPUFrameBuffer *fb = fbl->rtao_pos_norm_fb;
  
  GPU_framebuffer_bind(fb);
  GPU_framebuffer_read_color(
    fb, 0, 0,rtao_cpu_buff.w, rtao_cpu_buff.h, 3, 0, rtao_cpu_buff.norm);

  GPU_framebuffer_read_color(
    fb, 0, 0,rtao_cpu_buff.w, rtao_cpu_buff.h, 3, 1, rtao_cpu_buff.pos);

  // stop timer
  gettimeofday(&t_end, NULL);
  elapsed_time = t_end.tv_sec + t_end.tv_usec / 1e6 - t_start.tv_sec - t_start.tv_usec / 1e6; // in seconds

  printf("GPU_framebuffer read in %f seconds\n", elapsed_time);
}

struct GPUTexture;

void EEVEE_occlusion_trace_compute(EEVEE_ViewLayerData *UNUSED(sldata),
                             EEVEE_Data *vedata,
                             struct GPUTexture *depth_src,
                             int layer)
{
  printf("%s\n", "EEVEE_occlusion_trace_compute");
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  if ((effects->enabled_effects & EFFECT_GTAO) != 0) {
    DRW_stats_group_start("GTAO Trace Hits");
    effects->ao_src_depth = depth_src;
    
    if(!e_data.use_gpu_buff) {
      /* pure cpu pipeline */
      PVZ_occlusion_trace_build_prim_rays_cpu();
    } else {
      /* use gpu generated buffers for world positions and normals*/
      GPU_framebuffer_bind(fbl->rtao_pos_norm_fb);
      DRW_draw_pass(psl->ao_trace_pos);
      PVZ_read_gpu_buffers(vedata); // read world positions and normals from gpu
      PVZ_occlusion_trace_build_prim_rays_gpu();
    }

    PVZ_occlusion_trace_compute_embree();
    PVZ_hits_texture_update(effects->rtao_embree_tx_1, GPU_R8, rtao_cpu_buff.hits); // send ray hits back to gpu
    effects->rtao_embree_tx_final = effects->rtao_embree_tx_1;

    if(e_data.denoise) {
      struct GPUTexture *ao_tex; // just a pointer
      
      denoise_uniform_buffer_data.stepwidth = 1.0; // starting from smallest
      denoise_uniform_buffer_data.c_phi = rtao_deonise_data.c_phi;
      denoise_uniform_buffer_data.n_phi = rtao_deonise_data.n_phi;
      denoise_uniform_buffer_data.p_phi = rtao_deonise_data.p_phi;
      
      for(uint ii=0; ii<rtao_deonise_data.iterations; ii++) {
        DRW_uniformbuffer_update(e_data.denoise_ubo, &denoise_uniform_buffer_data);

        if (ii % 2 == 0) {
          // even iteration
          GPU_framebuffer_bind(fbl->rtao_denoise_fb_1); // writes to tx_2
          DRW_draw_pass(psl->ao_embree_denoise_pass1); // reads from raw
        } else {
          // odd iteration
          GPU_framebuffer_bind(fbl->rtao_denoise_fb_2); // writes to tx_1
          DRW_draw_pass(psl->ao_embree_denoise_pass2); // reads from final
        }
        denoise_uniform_buffer_data.stepwidth *= 2;
      }

      effects->rtao_embree_tx_final = (rtao_deonise_data.iterations % 2 == 0) ? effects->rtao_embree_tx_1 : effects->rtao_embree_tx_2;
    }

    /* Restore */
    GPU_framebuffer_bind(fbl->main_fb);

    DRW_stats_group_end();
  }
}

void EEVEE_occlusion_trace_draw_debug(EEVEE_ViewLayerData *UNUSED(sldata), EEVEE_Data *vedata)
{
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  if (((effects->enabled_effects & EFFECT_GTAO) != 0) && (G.debug_value == 6)) {
    DRW_stats_group_start("GTAO Debug");

    GPU_framebuffer_bind(fbl->rtao_pos_norm_fb);
    DRW_draw_pass(psl->ao_trace_pos);

    //GPU_framebuffer_bind(fbl->gtao_debug_fb);
    //DRW_draw_pass(psl->ao_embree_debug);

    /* Restore */
    GPU_framebuffer_bind(fbl->main_fb);

    DRW_stats_group_end();
  }
}

void EEVEE_occlusion_trace_output_accumulate(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata)
{
  printf("EEVEE_occlusion_trace_output_accumulate\n");
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_PassList *psl = vedata->psl;

  if (fbl->ao_accum_fb != NULL) {
    DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

    /* Update the min_max/horizon buffers so the refracion materials appear in it. */
    EEVEE_create_minmax_buffer(vedata, dtxl->depth, -1);
    EEVEE_occlusion_trace_compute(sldata, vedata, dtxl->depth, -1);

    GPU_framebuffer_bind(fbl->ao_accum_fb);
    DRW_draw_pass(psl->ao_accum_ps);

    /* Restore */
    GPU_framebuffer_bind(fbl->main_fb);
  }
}

void EEVEE_occlusion_trace_free(void)
{
  printf("EEVEE_occlusion_trace_free\n");
  
  e_data.embree_mode = false; // this might be needed to avoid unneccessary cache populate when engine switched

  DRW_SHADER_FREE_SAFE(e_data.gtao_sh);
  DRW_SHADER_FREE_SAFE(e_data.gtao_debug_sh);

  DRW_SHADER_FREE_SAFE(e_data.rtao_pos_norm_sh);
  DRW_SHADER_FREE_SAFE(e_data.rtao_denoise_sh);


  //BLI_memblock_clear(DST.vmempool->shgroups, e_data.shgrp);
  //MEM_SAFE_FREE(rtao_deonise_data.kernel);
  //MEM_SAFE_FREE(rtao_deonise_data.offset);
  
  DRW_UBO_FREE_SAFE(e_data.denoise_ubo);
  PVZ_occlusion_trace_buffers_free();
}
