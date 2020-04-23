#ifndef __EEVEE_OCLLUSION_TRACE_H__
#define __EEVEE_OCLLUSION_TRACE_H__

#include "eevee_embree.h"
#include "eevee_private.h"

void E_multDirMatrix(const EVEM_Matrix44f &mat, const EVEM_Vec3f &src, EVEM_Vec3f &dst); 

void PVZ_occlusion_trace_build_cpu_buffer(uint w, uint h);
void PVZ_occlusion_trace_build_prim_rays_buffer(uint w, uint h);
void PVZ_occlusion_trace_testfill_cpu_buffer(void);
void PVZ_texture_test(EEVEE_Data *vedata);

#endif // __EEVEE_OCLLUSION_TRACE_H__