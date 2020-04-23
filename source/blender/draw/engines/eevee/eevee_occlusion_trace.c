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

#include "BLI_string_utils.h"

#include "DEG_depsgraph_query.h"

#include "BKE_global.h" /* for G.debug_value */

#include "eevee_private.h"

#include "GPU_extensions.h"
#include "GPU_platform.h"
#include "GPU_state.h"
#include "GPU_texture.h"

#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "eevee_embree.h"
#include "eevee_occlusion_trace.h"

static struct {
  /* Ground Truth Ambient Occlusion */
  struct GPUShader *gtao_sh;
  struct GPUShader *gtao_debug_sh;

  struct GPUTexture *dummy_horizon_tx;
  Object *camera;
} e_data = {NULL}; /* Engine data */

static struct {
  uint w, h; /* cpu buffer width, height*/
  float *hits; /* embree hits buffer */
} ao_cpu_buff = {.hits = NULL, .w=0, .h=0}; /* CPU ao data */

extern struct EeveeEmbreeData evem_data;

static struct EeveeEmbreeRaysBuffer prim_rays_buff = {.rays=NULL, .rays16=NULL, .rays8=NULL, .rays4=NULL, .w=0, .h=0};

extern char datatoc_ambient_occlusion_trace_lib_glsl[];
extern char datatoc_common_view_lib_glsl[];
extern char datatoc_common_uniforms_lib_glsl[];
extern char datatoc_bsdf_common_lib_glsl[];
extern char datatoc_effect_gtao_trace_frag_glsl[];

static void eevee_create_shader_occlusion_trace(void)
{
  char *frag_str = BLI_string_joinN(datatoc_common_view_lib_glsl,
                                    datatoc_common_uniforms_lib_glsl,
                                    datatoc_bsdf_common_lib_glsl,
                                    datatoc_ambient_occlusion_trace_lib_glsl,
                                    datatoc_effect_gtao_trace_frag_glsl);

  e_data.gtao_sh = DRW_shader_create_fullscreen(frag_str, NULL);
  e_data.gtao_debug_sh = DRW_shader_create_fullscreen(frag_str, "#define DEBUG_AO\n");

  MEM_freeN(frag_str);
}

int EEVEE_occlusion_trace_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata, Object *camera)
{
  omp_set_num_threads(8);

  printf("%s\n", "EEVEE_occlusion_trace_init");
  printf("OpenMP num threads: %d\n", omp_get_num_threads());
  printf("OpenMP max threads: %d\n", omp_get_max_threads());
  e_data.camera = camera;

  EEVEE_CommonUniformBuffer *common_data = &sldata->common_data;
  EEVEE_FramebufferList *fbl = vedata->fbl;
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  if (!e_data.dummy_horizon_tx) {
    float pixel[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    e_data.dummy_horizon_tx = DRW_texture_create_2d(1, 1, GPU_R32F, DRW_TEX_WRAP, pixel);
  }

  if (scene_eval->eevee.flag & SCE_EEVEE_GTAO_ENABLED) {
    const float *viewport_size = DRW_viewport_size_get();
    const int fs_size[2] = {(int)viewport_size[0], (int)viewport_size[1]};

    /* Shaders */
    if (!e_data.gtao_sh) {
      eevee_create_shader_occlusion_trace();
    }

    common_data->ao_dist = scene_eval->eevee.gtao_distance;
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

    effects->gtao_trace_hits = DRW_texture_pool_query_2d(fs_size[0], fs_size[1], GPU_R32F, &draw_engine_eevee_type);
    GPU_framebuffer_ensure_config(&fbl->gtao_fb, {GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(effects->gtao_trace_hits)});

    if (G.debug_value == 6) {
      effects->gtao_horizons_debug = DRW_texture_pool_query_2d(
          fs_size[0], fs_size[1], GPU_RGBA8, &draw_engine_eevee_type);
      GPU_framebuffer_ensure_config(
          &fbl->gtao_debug_fb,
          {GPU_ATTACHMENT_NONE, GPU_ATTACHMENT_TEXTURE(effects->gtao_horizons_debug)});
    }
    else {
      effects->gtao_horizons_debug = NULL;
    }

    return EFFECT_GTAO | EFFECT_NORMAL_BUFFER;
  }

  /* Cleanup */
  effects->gtao_trace_hits = e_data.dummy_horizon_tx;
  GPU_FRAMEBUFFER_FREE_SAFE(fbl->gtao_fb);
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
    DRW_shgroup_uniform_texture_ref(grp, "ao_traceBuffer", &effects->gtao_trace_hits);
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
    DRW_PASS_CREATE(psl->ao_trace, DRW_STATE_WRITE_COLOR);
    DRWShadingGroup *grp = DRW_shgroup_create(e_data.gtao_sh, psl->ao_trace);
    DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
    DRW_shgroup_uniform_texture_ref(grp, "maxzBuffer", &txl->maxzbuffer);
    DRW_shgroup_uniform_texture_ref(grp, "normalBuffer", &effects->ssr_normal_input);
    DRW_shgroup_uniform_texture_ref(grp, "ao_traceBuffer", &effects->gtao_trace_hits);
    DRW_shgroup_uniform_block(grp, "common_block", sldata->common_ubo);
    DRW_shgroup_uniform_block(
        grp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));
    DRW_shgroup_call(grp, quad, NULL);

    if (G.debug_value == 6) {
      DRW_PASS_CREATE(psl->ao_horizon_debug, DRW_STATE_WRITE_COLOR);
      grp = DRW_shgroup_create(e_data.gtao_debug_sh, psl->ao_horizon_debug);
      DRW_shgroup_uniform_texture(grp, "utilTex", EEVEE_materials_get_util_tex());
      DRW_shgroup_uniform_texture_ref(grp, "maxzBuffer", &txl->maxzbuffer);
      DRW_shgroup_uniform_texture_ref(grp, "depthBuffer", &dtxl->depth);
      DRW_shgroup_uniform_texture_ref(grp, "normalBuffer", &effects->ssr_normal_input);
      DRW_shgroup_uniform_texture_ref(grp, "ao_traceBuffer", &effects->ao_src_depth);
      DRW_shgroup_uniform_block(grp, "common_block", sldata->common_ubo);
      DRW_shgroup_uniform_block(
          grp, "renderpass_block", EEVEE_material_default_render_pass_ubo_get(sldata));
      DRW_shgroup_call(grp, quad, NULL);
    }
  }
}

void PVZ_occlusion_trace_build_cpu_buffer(uint w, uint h) {
  /* buffer exist, no size changes*/
  if((ao_cpu_buff.hits != NULL) && ao_cpu_buff.w == w && ao_cpu_buff.h == h) {
    printf("%s\n", "reuse cpu buff");
    return;
  }

  /* viewport size changed */
  if(ao_cpu_buff.hits != NULL) {
    printf("%s\n", "free cpu buff");
    MEM_freeN(ao_cpu_buff.hits);
  }  

  ao_cpu_buff.w = w;
  ao_cpu_buff.h = h;
  
  /* (re)create it */
  printf("%s\n", "create cpu buff");
  ao_cpu_buff.hits = MEM_mallocN(sizeof(float) * w * h, "ao_hits_buffer");
  printf("%s\n", "cpu buff created ");
}

/* primary rays buffer */
void PVZ_occlusion_trace_build_prim_rays_buffer(uint w, uint h) {
  if (prim_rays_buff.w == w && prim_rays_buff.h == h)return; // buffer size not changed

  EVEM_rays_buffer_free(&prim_rays_buff); // free previous buffer

  prim_rays_buff.rays = MEM_mallocN(sizeof(struct RTCRay) * w * h , "prim rays buff");

  prim_rays_buff.w = w;
  prim_rays_buff.h = h;
}

void PVZ_occlusion_trace_testfill_cpu_buffer(void) {
  printf("%s\n", "fill");
  #pragma omp parallel for num_threads(8) collapse(2)
  for( int x=0; x < ao_cpu_buff.w; x++){
    for( int y=0; y < ao_cpu_buff.h; y++) {
      ao_cpu_buff.hits[x + y*ao_cpu_buff.w] = (x % 255) / 255.0;
    }
  }
  printf("%s\n", "filled");
}

void E_multDirMatrix(const EVEM_Matrix44f &mat, const EVEM_Vec3f &src, EVEM_Vec3f &dst) const {
  dst.x = src[0] * mat.x[0][0] + src[1] * mat.x[1][0] + src[2] * mat.x[2][0]; 
  dst.y = src[0] * mat.x[0][1] + src[1] * mat.x[1][1] + src[2] * mat.x[2][1]; 
  dst.z = src[0] * mat.x[0][2] + src[1] * mat.x[1][2] + src[2] * mat.x[2][2]; 
}

void PVZ_occlusion_trace_test_trace_occlusion(void) {
  printf("%s\n", "test trace occlusion");
  struct timeval t_start, t_end;
  double elapsed_time;

  // start timer
  gettimeofday(&t_start, NULL);

  struct RTCIntersectContext context;

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  rtcInitIntersectContext(&prim_rays_buff.context);

  struct RTCRay *rays = prim_rays_buff.rays;

  uint width = ao_cpu_buff.w, height = ao_cpu_buff.h;
  float scale = 1.0;
  float image_aspect_ratio = width / (float)height;

  struct EVEM_Matrix44f xfm;// = {1.0};
  uint ix, iy; // pixel pos
  EVEM_Vec3f v = {-1.0};
  
  #pragma omp parallel for num_threads(8) private(ix, iy, v)
  for( int i=0; i < ao_cpu_buff.w * ao_cpu_buff.h; i++){
      rays[i].id = i;
      ix = i - ((i / ao_cpu_buff.w) * ao_cpu_buff.w);
      iy = i / ao_cpu_buff.w;

      v.x = (2 * (ix + 0.5) / (float)width - 1) * scale; 
      v.y = (1 - 2 * (iy + 0.5) / (float)height) * scale * 1 / image_aspect_ratio;

      EVEM_Vec3f dir;
      E_multDirMatrix(xfm, v, dir);
      // normalize(dir); 

      rays[i].mask = 0xFFFFFFFF;
      rays[i].org_x = -1000.0;
      rays[i].org_z = v.x;
      rays[i].org_y = v.y;
      rays[i].tnear = 0.0;

      rays[i].dir_x = 1.0;
      rays[i].dir_y = 0.0;
      rays[i].dir_z = 0.0;
      rays[i].tfar = 100000.0;
      rays[i].time = 0.0;
          
      //rtcOccluded1(evem_data.scene, &prim_rays_buff.context, &ray);
      //ao_cpu_buff.hits[i] = ray.tfar;
      //ao_cpu_buff.hits[i] = (ray.tfar < 0.0f) ? 0.0 : 1.0;
  }
  
  //rtcOccluded1M(evem_data.scene, &prim_rays_buff.context, (struct RTCRay*)&prim_rays_buff.rays, 16, sizeof(struct RTCRay));

  #pragma omp parallel for num_threads(8)
  for( int i=0; i < ao_cpu_buff.w * ao_cpu_buff.h; i++){
    ao_cpu_buff.hits[rays[i].id] = rays[i].org_z;
  }
  
  // stop timer
  gettimeofday(&t_end, NULL);
  elapsed_time = t_end.tv_sec + t_end.tv_usec / 1e6 - t_start.tv_sec - t_start.tv_usec / 1e6; // in seconds

  printf("test trace occlusion done in %f seconds with %d rays \n", elapsed_time, ao_cpu_buff.w*ao_cpu_buff.h);
}

void PVZ_texture_test(EEVEE_Data *vedata) {
  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  printf("NORMAL TEXTURE WIDTH %d\n", GPU_texture_width(&effects->ssr_normal_input));
  printf("NORMAL TEXTURE HEIGHT %d\n", GPU_texture_height(&effects->ssr_normal_input));

  //short *tex_data = (short *)GPU_texture_read(&effects->ssr_normal_input, GPU_RG16, 0);

  //if (!tex_data) {
  //  printf("%s\n", "GPU_texture_read ERROR !!!!!!!!!!!");
  //  return;
  //}

  uint i;
  for( int x=0; x < ao_cpu_buff.w; x++){
    for( int y=0; y < ao_cpu_buff.h; y++) {
      i = x + y*ao_cpu_buff.w;
      //ao_cpu_buff.hits[i] = tex_data[i];
    }
  }

  //MEM_freeN(tex_data);
}

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
    //GPU_framebuffer_bind(fbl->main_fb);

    //DRW_draw_pass(psl->ao_trace);

    const float *viewport_size = DRW_viewport_size_get();
    const int fs_size[2] = {(int)viewport_size[0], (int)viewport_size[1]};

    PVZ_occlusion_trace_build_cpu_buffer(fs_size[0], fs_size[1]);
    PVZ_occlusion_trace_build_prim_rays_buffer(fs_size[0], fs_size[1]);
    PVZ_occlusion_trace_test_trace_occlusion();
    //PVZ_texture_test(vedata);

    GPU_texture_update(effects->gtao_trace_hits, GPU_R32F, ao_cpu_buff.hits);

    /* Restore */
    //GPU_framebuffer_bind(fbl->main_fb);

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

    GPU_framebuffer_bind(fbl->gtao_debug_fb);
    DRW_draw_pass(psl->ao_horizon_debug);

    /* Restore */
    GPU_framebuffer_bind(fbl->main_fb);

    DRW_stats_group_end();
  }
}

void EEVEE_occlusion_trace_output_accumulate(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata)
{
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
  DRW_SHADER_FREE_SAFE(e_data.gtao_sh);
  DRW_SHADER_FREE_SAFE(e_data.gtao_debug_sh);
  DRW_TEXTURE_FREE_SAFE(e_data.dummy_horizon_tx);
  MEM_freeN(ao_cpu_buff.hits);

  EVEM_rays_buffer_free(&prim_rays_buff);
}
