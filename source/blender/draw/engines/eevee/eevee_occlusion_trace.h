#ifndef __EEVEE_OCLLUSION_TRACE_H__
#define __EEVEE_OCLLUSION_TRACE_H__

#include "eevee_private.h"


void PVZ_occlusion_trace_build_cpu_buffer(uint w, uint h);
void PVZ_occlusion_trace_testfill_cpu_buffer(void);
void PVZ_texture_test(EEVEE_Data *vedata);

#endif // __EEVEE_OCLLUSION_TRACE_H__