#ifndef __EEVEE_OCLLUSION_TRACE_H__
#define __EEVEE_OCLLUSION_TRACE_H__

#include <embree3/rtcore.h>

#include "eevee_embree.h"
#include "eevee_private.h"

typedef struct embree_trace_ray_stream_args {
   uint worker_id;
   uint job_id;
   RTCScene *scene;
   struct RTCIntersectContext *context;
   struct RTCRay *rays;
   uint stream_size;
}EmbreeTraceRayStreamWorkerData;

void embree_trace_ray_stream_worker(void *arg);

void PVZ_occlusion_trace_buffers_init(uint w, uint h);
void PVZ_occlusion_trace_compute_embree(void);
void PVZ_occlusion_trace_build_prim_rays_cpu(void);
void PVZ_occlusion_trace_build_prim_rays_gpu(void);
void PVZ_read_gpu_buffers(EEVEE_Data *vedata);
void PVZ_occlusion_trace_buffers_free(void);

#endif // __EEVEE_OCLLUSION_TRACE_H__