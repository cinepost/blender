#ifndef __EEVEE_OCLLUSION_TRACE_H__
#define __EEVEE_OCLLUSION_TRACE_H__

#include "eevee_embree.h"
#include "eevee_private.h"


void PVZ_occlusion_trace_buffers_init(uint w, uint h);
void PVZ_occlusion_trace_compute_embree(void);
void PVZ_read_gpu_buffers(EEVEE_Data *vedata);
void PVZ_occlusion_trace_buffers_free(void);

#endif // __EEVEE_OCLLUSION_TRACE_H__