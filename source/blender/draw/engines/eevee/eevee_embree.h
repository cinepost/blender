#ifndef __EEVEE_EMBREE_H__
#define __EEVEE_EMBREE_H__

#include <embree3/rtcore.h>
#include <embree3/rtcore_scene.h>

#include "BKE_object.h"
#include "DEG_depsgraph_query.h"
#include "DNA_world_types.h"
#include "DNA_mesh_types.h"

#include "eevee_private.h"
#include "eevee_embree_objects.h"

#ifdef  __cplusplus
extern "C" {
#endif

/* size of screen tiles */
#define EMBREE_TILE_SIZE_X 8
#define EMBREE_TILE_SIZE_Y 8

/* vertex and triangle layout */
struct EVEM_Vertex3f { float x,y,z; };
typedef struct EVEM_Vertex3f EVEM_Vertex3f;

struct EVEM_Triangle { int v0, v1, v2; };
typedef struct EVEM_Triangle EVEM_Triangle;

//struct EVEM_Vec3f { float x,y,z; };
//typedef struct EVEM_Vec3f EVEM_Vec3f;
typedef float EVEM_Vec3f[3];
typedef double EVEM_Vec3d[3];

//struct EVEM_Matrix44f { float x[4][4]; };
//typedef struct EVEM_Matrix44f EVEM_Matrix44f;
typedef float EVEM_Matrix44f[4][4];
typedef double EVEM_Matrix44d[4][4];


/* embree rays(packets) buffer*/
struct EeveeEmbreeRaysBuffer {
	struct RTCRay16 *rays16;
	struct RTCRay8  *rays8;
	struct RTCRay4  *rays4;
	struct RTCRay   *rays;

	struct RTCIntersectContext context;
	uint w;
	uint h;
};

/* global embree data */
struct EeveeEmbreeData {
  RTCDevice device; /* embree device */
  RTCScene scene;   /* embree scene */

  // stats and capabilities
  bool NATIVE_RAY4_ON, NATIVE_RAY8_ON, NATIVE_RAY16_ON;
  bool RAY_STREAM_ON;
  uint TASKING_SYSTEM;

  bool update_tlas; // top level acceleration structure need to be updated
  bool update_blas; // bottom level acceleration structure need to be updated

  // eevee related stuff
  float sample_num;
  bool  embree_enabled;
};


/* Macro's */
#define pvz_max(a,b) \
   ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
      _a > _b ? _a : _b; })

/* Functions */
void EEVEE_embree_init(EEVEE_ViewLayerData *sldata);
void EEVEE_embree_print_capabilities(void);
void EEVEE_embree_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata);
void EEVEE_embree_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool cast_shadow);
void EEVEE_embree_cache_finish(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata);

void EEVEE_embree_free(void);

void EVEM_create_object(Object *ob, ObjectInfo *ob_info);
void EVEM_update_object(Object *ob, ObjectInfo *ob_info);

void EVEM_mesh_object_clear(Mesh *me);
void EVEM_mesh_object_create(Mesh *me, ObjectInfo *ob_info);
void EVEM_mesh_object_update(Object *ob, ObjectInfo *ob_info);
void EVEM_instance_update_transform(Object *ob, ObjectInfo *ob_info);
void EVEM_toggle_object_visibility(Object *ob, ObjectInfo *ob_info);

void EVEM_rays_buffer_free(struct EeveeEmbreeRaysBuffer *buff);

//inline EVEM_Vec3f EVEM_mult_dir_matrix(EVEM_Matrix44f mat, EVEM_Vec3f vec);

#ifdef  __cplusplus
}
#endif

#endif // __EEVEE_EMBREE_H__