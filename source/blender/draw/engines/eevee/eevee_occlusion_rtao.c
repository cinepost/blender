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

#ifndef _WIN32
#include <unistd.h>
#include <sys/time.h>
#endif

#include <math.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "pthread.h"

#include "debug.h"
#include "eevee_embree.h"
#include "eevee_occlusion_rtao.h"

typedef struct GPUTexture GPUTexture;

#define NUM_THREADS 8
#define RAYS_STREAM_SIZE 64
#define EMBREE_TRACE_MIN_BIAS 0.0002f

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

  struct GPUTexture *dummy_tx;
  struct GPUTexture *normals_tx; // normals texture pointer copy. on first sample we use own normals, other passes use ssr_normals

  Object *camera;
  float ao_dist;
  float gpu_bias;
  bool denoise;
  bool embree_mode;
  float sample_num;
  float ao_use_bump;
} e_data = {NULL, .denoise = false, .gpu_bias = 0.005, .embree_mode = true, .sample_num = 1.0f, .ao_use_bump = 0.0f}; /* Engine data */

static struct {
  uint w, h; /* cpu buffer width, height*/
  struct RTCRay *rays;
  unsigned char *hits; /* embree occlusion hits buffer */ // TODO: try pack 4or8 hits in one byte
  float *nrm;   /* world normal cpu buffer*/
  float *pos;    /* world posisio */

  float *nrm_;   /* world normal cpu buffer*/
  float *pos_;    /* world posisio */

} rtao_cpu_buffers = {.hits = NULL, .w=0, .h=0}; /* CPU ao data */

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
  
  e_data.rtao_pos_norm_sh = GPU_shader_create(vert_str, frag_str, NULL, NULL, "#define AO_TRACE_POS\n", "rtao_nd_shader");//__func__);

  e_data.gtao_debug_sh = DRW_shader_create_fullscreen(frag_str, "#define DEBUG_AO\n");
  e_data.rtao_denoise_sh = DRW_shader_create_fullscreen(denoise_frag_str, NULL);

  MEM_freeN(frag_str);
  MEM_freeN(vert_str);
  MEM_freeN(denoise_frag_str);
}

int EEVEE_occlusion_trace_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata, Object *camera)
{
  dbg_printf("%s\n", "EEVEE_occlusion_trace_init");
  omp_set_num_threads(16);
  dbg_printf("OpenMP num threads: %d\n", omp_get_num_threads());
  dbg_printf("OpenMP max threads: %d\n", omp_get_max_threads());
  e_data.camera = camera;

  EEVEE_CommonUniformBuffer *common_data = &sldata->common_data;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;
  DefaultTextureList *dtxl = DRW_viewport_texture_list_get();

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  if (!e_data.dummy_tx) {
    float pixel[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    e_data.dummy_tx = DRW_texture_create_2d(1, 1, GPU_RGBA8, DRW_TEX_WRAP, pixel);
  }

  e_data.denoise = (scene_eval->eevee.flag & SCE_EEVEE_RTAO_DENOISE) ? true : false;

  e_data.embree_mode = false;
  common_data->ao_embree = 0.0;
  common_data->ao_use_bump = 0.0;

  if ((scene_eval->eevee.flag & SCE_EEVEE_GTAO_ENABLED) && (scene_eval->eevee.flag & SCE_EEVEE_RTAO_ENABLED)) {
    e_data.embree_mode = true;

    const float *viewport_size = DRW_viewport_size_get();
    const int fs_size[2] = {(int)viewport_size[0], (int)viewport_size[1]};

    /* UI data fetch */
    rtao_deonise_data.iterations = pvz_max(1, (uint)scene_eval->eevee.rtao_denoise_iterations);
    rtao_deonise_data.c_phi = scene_eval->eevee.rtao_denoise_c_phi;
    rtao_deonise_data.n_phi = scene_eval->eevee.rtao_denoise_n_phi;
    rtao_deonise_data.p_phi = scene_eval->eevee.rtao_denoise_p_phi;

    e_data.gpu_bias = scene_eval->eevee.rtao_gpubuff_bias;

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

    effects->rtao_bent_normal = e_data.dummy_tx;

    /* Common ubo data */
    common_data->ao_embree = 1.0f;
    e_data.ao_dist = common_data->ao_dist / 2.0;

    if (scene_eval->eevee.flag & SCE_EEVEE_RTAO_BUMP) {
      common_data->ao_use_bump = 1.0f; /* USE_BUMP */
    }

    bool taa_use_reprojection = (stl->effects->enabled_effects & EFFECT_TAA_REPROJECT) != 0;
    if (DRW_state_is_image_render() || taa_use_reprojection || ((stl->effects->enabled_effects & EFFECT_TAA) != 0)) {
      evem_data.sample_num = e_data.sample_num = taa_use_reprojection ? stl->effects->taa_reproject_sample + 1 : stl->effects->taa_current_sample;
    }

    effects->rtao_embree_tx_1 = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_R8, &draw_engine_eevee_type);
    effects->rtao_embree_tx_2 = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_R8, &draw_engine_eevee_type);

    effects->rtao_nrm = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_RGB16F, &draw_engine_eevee_type);
    effects->rtao_pos = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_RGB32F, &draw_engine_eevee_type);
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

  /* dummy textures */
  effects->rtao_embree_tx_1 = e_data.dummy_tx;
  effects->rtao_embree_tx_2 = e_data.dummy_tx;
  effects->rtao_nrm = e_data.dummy_tx;
  effects->rtao_pos = e_data.dummy_tx;
  effects->rtao_embree_tx_final = e_data.dummy_tx;
  effects->rtao_bent_normal = e_data.dummy_tx;

  /* Cleanup */
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_pos_norm_fb);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_denoise_fb_1);
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->rtao_denoise_fb_2);

  return 0;
}

void EEVEE_occlusion_trace_output_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata, uint tot_samples)
{
  dbg_printf("%s\n", "EEVEE_occlusion_trace_output_init");
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
    DRW_shgroup_uniform_texture_ref(grp, "embreeHitsBuffer", &effects->rtao_embree_tx_final);
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
  dbg_printf("%s\n", "EEVEE_occlusion_trace_cache_init");
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

    /* This pass is used to calculate ao rays positions on cpu */
    DRW_PASS_CREATE(psl->ao_trace_pos, DRW_STATE_WRITE_COLOR | DRW_STATE_WRITE_DEPTH | DRW_STATE_DEPTH_EQUAL | DRW_STATE_CLIP_PLANES);// | DRW_STATE_CULL_BACK);
    stl->g_data->rtao_shgrp = DRW_shgroup_create(e_data.rtao_pos_norm_sh, (DRWPass *)psl->ao_trace_pos);
    DRW_shgroup_uniform_texture(stl->g_data->rtao_shgrp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_block(stl->g_data->rtao_shgrp, "common_block", sldata->common_ubo);
    DRW_shgroup_uniform_float(stl->g_data->rtao_shgrp, "sampleNum", &e_data.sample_num, 1);
    DRW_shgroup_uniform_float(stl->g_data->rtao_shgrp, "aoUseBump", &e_data.ao_use_bump, 1);
    //DRW_shgroup_uniform_texture_ref(e_data.shgrp, "depthBuffer", &effects->ao_src_depth);
    DRW_shgroup_uniform_texture_ref(stl->g_data->rtao_shgrp, "normalBuffer", &effects->ssr_normal_input); // use on samples starting from 1
    DRW_shgroup_uniform_block(
        stl->g_data->rtao_shgrp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));

    /* Desnoising passese and debug */
    DRWShadingGroup *grp = NULL;
    if (e_data.denoise) {
      /* two passes needed for several denoising iterations */
      DRW_PASS_CREATE(psl->ao_embree_denoise_pass1, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.rtao_denoise_sh, psl->ao_embree_denoise_pass1);
      DRW_shgroup_uniform_texture_ref(grp, "wposBuffer",  &effects->rtao_pos);
      DRW_shgroup_uniform_texture_ref(grp, "wnormBuffer", &effects->rtao_nrm);
      DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeRawBuffer", &effects->rtao_embree_tx_1);
      DRW_shgroup_uniform_float(grp, "sampleNum", & e_data.sample_num, 1);
      //DRW_shgroup_uniform_ivec2(grp, "offset", &, 50);
      //DRW_shgroup_uniform_float(grp, "kernel", &, 25);
      DRW_shgroup_uniform_block(grp, "denoise_block", e_data.denoise_ubo);
      DRW_shgroup_call(grp, quad, NULL);

      DRW_PASS_CREATE(psl->ao_embree_denoise_pass2, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.rtao_denoise_sh, psl->ao_embree_denoise_pass2);
      DRW_shgroup_uniform_texture_ref(grp, "wposBuffer",  &effects->rtao_pos);
      DRW_shgroup_uniform_texture_ref(grp, "wnormBuffer", &effects->rtao_nrm);
      DRW_shgroup_uniform_texture_ref(grp, "aoEmbreeRawBuffer", &effects->rtao_embree_tx_2);
      DRW_shgroup_uniform_float(grp, "sampleNum", & e_data.sample_num, 1);
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

void EEVEE_rtao_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool cast_shadow) {
  EEVEE_StorageList *stl = vedata->stl;
  if(!stl->g_data->rtao_shgrp || !e_data.embree_mode) return;
  //dbg_printf("%s\n", "EEVEE_rtao_cache_populate");
  //if(!cast_shadow) return; // don't draw pos/norm buffer for non shadow casters

  struct GPUBatch *geom = NULL;
  geom = DRW_cache_object_surface_get(ob);
  DRW_shgroup_call(stl->g_data->rtao_shgrp, geom, ob);
}

/*
 * Builds buffers that are necessary for our occlusion ray tracing
 */
void PVZ_occlusion_trace_buffers_init(uint w, uint h) {
  /* buffers exist and no size changes required */
  if((rtao_cpu_buffers.hits != NULL) && rtao_cpu_buffers.w == w && rtao_cpu_buffers.h == h) {
    return;
  }

  /* viewport size changed */
  if(rtao_cpu_buffers.hits != NULL) {
     PVZ_occlusion_trace_buffers_free();
  }  

  rtao_cpu_buffers.w = w;
  rtao_cpu_buffers.h = h;
  
  /* (re)create buffers  */
  rtao_cpu_buffers.hits = (unsigned char *)malloc(sizeof(unsigned char) * w * h);
  rtao_cpu_buffers.nrm = malloc(sizeof(float) * w * h * 3);
  rtao_cpu_buffers.pos = malloc(sizeof(float) * w * h * 3);
  rtao_cpu_buffers.rays = (struct RTCRay *)malloc(sizeof(struct RTCRay) * w * h);
}

void PVZ_occlusion_trace_buffers_free(void) {
  rtao_cpu_buffers.w = 0;
  rtao_cpu_buffers.h = 0;
  if(rtao_cpu_buffers.hits) free(rtao_cpu_buffers.hits);
  if(rtao_cpu_buffers.nrm) free(rtao_cpu_buffers.nrm);
  if(rtao_cpu_buffers.pos) free(rtao_cpu_buffers.pos);
  if(rtao_cpu_buffers.rays) free(rtao_cpu_buffers.rays);
}

#pragma GCC push_options
#pragma GCC optimize ("unroll-loops")


void PVZ_occlusion_trace_build_prim_rays_gpu(void) {
  struct RTCRay *rays = rtao_cpu_buffers.rays;
  float ray_bias = pvz_max(EMBREE_TRACE_MIN_BIAS, e_data.gpu_bias);

  int i;
  #pragma omp parallel for shared(rays, ray_bias) private(i)
  for(i=0; i < rtao_cpu_buffers.w * rtao_cpu_buffers.h; i++){
    rays[i].id = i; // we need this as Embree might rearrange rays for better performance

    rays[i].mask = 0xFFFFFFFF;
    rays[i].org_x = rtao_cpu_buffers.pos[i*3];
    rays[i].org_y = rtao_cpu_buffers.pos[i*3+1];
    rays[i].org_z = rtao_cpu_buffers.pos[i*3+2];
    rays[i].tnear = ray_bias;

    rays[i].dir_x = rtao_cpu_buffers.nrm[i*3];
    rays[i].dir_y = rtao_cpu_buffers.nrm[i*3+1];
    rays[i].dir_z = rtao_cpu_buffers.nrm[i*3+2];
    rays[i].tfar = e_data.ao_dist;
    rays[i].time = 0.0f;
  }
}

void PVZ_occlusion_trace_compute_embree(void) {
  dbg_printf("%s\n", "embree trace occlusion");

  #ifndef _WIN32
  // start timer
  struct timeval t_start, t_end;
  double elapsed_time;
  gettimeofday(&t_start, NULL);
  #endif

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);


  struct RTCIntersectContext ctx;

  rtcInitIntersectContext(&ctx);
  ctx.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

  struct RTCRay *rays = rtao_cpu_buffers.rays;

  uint num_pixels = rtao_cpu_buffers.w * rtao_cpu_buffers.h;

  int i;
  RTCScene scene = evem_data.scene;
  #pragma omp parallel for shared(scene, rays, ctx, rtao_cpu_buffers) private(i)
  for(i=0; i < num_pixels; i+=RAYS_STREAM_SIZE) {
    rtcOccluded1M(scene, &ctx, &rays[i], RAYS_STREAM_SIZE, sizeof(struct RTCRay));
    
    struct RTCRay *ray = NULL;
    
    #ifndef _WIN32
    #pragma omp simd
    #endif

    for(uint j=0; j < RAYS_STREAM_SIZE; j++){
      ray = &rays[i + j];
      rtao_cpu_buffers.hits[ray->id] = (ray->tfar >= 0.0f) ? 255 : 0;
    }
  }
  
  #ifndef _WIN32
  // stop timer
  gettimeofday(&t_end, NULL);
  elapsed_time = t_end.tv_sec + t_end.tv_usec / 1e6 - t_start.tv_sec - t_start.tv_usec / 1e6; // in seconds
  dbg_printf("test trace occlusion done in %f seconds with %u rays \n", elapsed_time, rtao_cpu_buffers.w*rtao_cpu_buffers.h);
  #endif
}

void EEVEE_occlusion_rtao_read_gpu_buffers(EEVEE_Data *vedata) {
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_EffectsInfo *effects = stl->effects;

  GPUFrameBuffer *fb = fbl->rtao_pos_norm_fb;
  
  GPU_framebuffer_bind(fb);
  GPU_framebuffer_read_color(fb, 0, 0,rtao_cpu_buffers.w, rtao_cpu_buffers.h, 3, 0, rtao_cpu_buffers.nrm);
  GPU_framebuffer_read_color(fb, 0, 0,rtao_cpu_buffers.w, rtao_cpu_buffers.h, 3, 1, rtao_cpu_buffers.pos);

  PVZ_occlusion_trace_build_prim_rays_gpu();
}

void EEVEE_occlusion_rtao_read_gpu_buffers_fast(EEVEE_Data *vedata) {
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_EffectsInfo *effects = stl->effects;

  struct RTCRay *rays = rtao_cpu_buffers.rays;
  float ray_bias = pvz_max(EMBREE_TRACE_MIN_BIAS, e_data.gpu_bias);

  GPU_framebuffer_bind(fbl->rtao_pos_norm_fb);
  
  uint pbo_buff_size = rtao_cpu_buffers.w * rtao_cpu_buffers.h * 3 * sizeof(float);
 
  GLuint pbo[2];
  glGenBuffers(1, &pbo[0]);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
  glBufferData(GL_PIXEL_PACK_BUFFER, pbo_buff_size, NULL, GL_STREAM_COPY);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  glGenBuffers(1, &pbo[1]);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
  glBufferData(GL_PIXEL_PACK_BUFFER, pbo_buff_size, NULL, GL_STREAM_COPY);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
  glReadPixels(0, 0, rtao_cpu_buffers.w, rtao_cpu_buffers.h, GL_RGB, GL_FLOAT, 0);

  glReadBuffer(GL_COLOR_ATTACHMENT1);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
  glReadPixels(0, 0, rtao_cpu_buffers.w, rtao_cpu_buffers.h, GL_RGB, GL_FLOAT, 0);

  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[0]);
  rtao_cpu_buffers.nrm_ = (float *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo[1]);
  rtao_cpu_buffers.pos_ = (float *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);
  if(rtao_cpu_buffers.nrm_) {
    int i;
    #pragma omp parallel for shared(rays, ray_bias) private(i)
    for(i=0; i < rtao_cpu_buffers.w * rtao_cpu_buffers.h; i++){
      rays[i].id = i; // we need this as Embree might rearrange rays for better performance
      rays[i].mask = 0xFFFFFFFF;
      rays[i].dir_x = rtao_cpu_buffers.nrm_[i*3];
      rays[i].dir_y = rtao_cpu_buffers.nrm_[i*3+1];
      rays[i].dir_z = rtao_cpu_buffers.nrm_[i*3+2];
      rays[i].tnear = ray_bias;
      rays[i].tfar = e_data.ao_dist;
      rays[i].time = 0.0f;

      rays[i].org_x = rtao_cpu_buffers.pos_[i*3];
      rays[i].org_y = rtao_cpu_buffers.pos_[i*3+1];
      rays[i].org_z = rtao_cpu_buffers.pos_[i*3+2];
    }
  }
  glUnmapBuffer(GL_PIXEL_PACK_BUFFER);

  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glDeleteBuffers(1, &pbo[0]);
  glDeleteBuffers(1, &pbo[1]);
}

#ifndef _WIN32
#pragma GCC pop_options
#endif

void EEVEE_occlusion_rtao_texture_update(GPUTexture *tex, eGPUDataFormat data_format, const void *pixels) {
  GLint alignment;
  glGetIntegerv(GL_UNPACK_ALIGNMENT, &alignment);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glBindTexture(tex->target, tex->bindcode);
  glTexSubImage2D(tex->target, 0, 0, 0, tex->w, tex->h, GL_RED, GL_UNSIGNED_BYTE, pixels);

  glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);
  glBindTexture(tex->target, 0);
}

void EEVEE_occlusion_trace_compute(EEVEE_ViewLayerData *UNUSED(sldata),
                             EEVEE_Data *vedata,
                             struct GPUTexture *depth_src,
                             int layer)
{
  dbg_printf("%s\n", "EEVEE_occlusion_trace_compute");
  EEVEE_PassList *psl = vedata->psl;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  if ( effects->enabled_effects & EFFECT_GTAO ) {
    DRW_stats_group_start("GTAO Trace Hits");
    effects->ao_src_depth = depth_src;
    
    /* use gpu generated buffers for world positions and normals*/
    GPU_framebuffer_bind(fbl->rtao_pos_norm_fb);

    DRW_draw_pass((DRWPass *)psl->ao_trace_pos);
    
    #ifndef _WIN32
    struct timeval t_start, t_end;
    double elapsed_time;
    // start timer
    gettimeofday(&t_start, NULL);
    #endif

    //EEVEE_occlusion_rtao_read_gpu_buffers(vedata);
    EEVEE_occlusion_rtao_read_gpu_buffers_fast(vedata); // read world positions and normals from gpu

    #ifndef _WIN32
    // stop timer
    gettimeofday(&t_end, NULL);
    elapsed_time = t_end.tv_sec + t_end.tv_usec / 1e6 - t_start.tv_sec - t_start.tv_usec / 1e6; // in seconds
    dbg_printf("GPU_framebuffer FULL read in %f seconds\n", elapsed_time);
    #endif

    PVZ_occlusion_trace_compute_embree();
    EEVEE_occlusion_rtao_texture_update(effects->rtao_embree_tx_1, GPU_R8, rtao_cpu_buffers.hits); // send ray hits back to gpu
    effects->rtao_embree_tx_final = effects->rtao_embree_tx_1;

    if(e_data.denoise) {
      
      denoise_uniform_buffer_data.stepwidth = 1.0; // starting from smallest
      denoise_uniform_buffer_data.c_phi = rtao_deonise_data.c_phi;
      denoise_uniform_buffer_data.n_phi = rtao_deonise_data.n_phi;
      denoise_uniform_buffer_data.p_phi = rtao_deonise_data.p_phi;
      
      for(uint i=0; i<rtao_deonise_data.iterations; i++) {
        DRW_uniformbuffer_update(e_data.denoise_ubo, &denoise_uniform_buffer_data);

        if (i % 2 == 0) {
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
    DRW_draw_pass((DRWPass *)psl->ao_trace_pos);

    //GPU_framebuffer_bind(fbl->gtao_debug_fb);
    //DRW_draw_pass(psl->ao_embree_debug);

    /* Restore */
    GPU_framebuffer_bind(fbl->main_fb);

    DRW_stats_group_end();
  }
}

void EEVEE_occlusion_trace_output_accumulate(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata)
{
  dbg_printf("EEVEE_occlusion_trace_output_accumulate\n");
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
  dbg_printf("EEVEE_occlusion_trace_free\n");
  
  e_data.embree_mode = false; // this might be needed to avoid unneccessary cache populate when engine switched

  DRW_SHADER_FREE_SAFE(e_data.gtao_sh);
  DRW_SHADER_FREE_SAFE(e_data.gtao_debug_sh);

  DRW_SHADER_FREE_SAFE(e_data.rtao_pos_norm_sh);
  DRW_SHADER_FREE_SAFE(e_data.rtao_denoise_sh);

  DRW_TEXTURE_FREE_SAFE(e_data.dummy_tx);
  
  DRW_UBO_FREE_SAFE(e_data.denoise_ubo);
  PVZ_occlusion_trace_buffers_free();
}
