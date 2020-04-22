#ifndef __EEVEE_EMBREE_H__
#define __EEVEE_EMBREE_H__

#include <embree3/rtcore.h>
#include <embree3/rtcore_scene.h>

#include "BKE_object.h"
#include "DEG_depsgraph_query.h"
#include "DNA_world_types.h"
#include "DNA_mesh_types.h"

#include "eevee_private.h"

#ifdef  __cplusplus
extern "C" {
#endif

/* size of screen tiles */
#define EMBREE_TILE_SIZE_X 8
#define EMBREE_TILE_SIZE_Y 8

/* vertex and triangle layout */
struct EVEM_Vertex3f { float x,y,z;  };
typedef struct EVEM_Vertex3f EVEM_Vertex3f;
struct EVEM_Triangle { int v0, v1, v2; };
typedef struct EVEM_Triangle EVEM_Triangle;


struct EeveeEmbreeRaysBuffer {
	struct RTCRay16 *ray16;
	struct RTCRay8  *ray8;
	struct RTCRay4  *ray4;
	struct RTCRay   *ray;

	struct RTCIntersectContext context;
};

/* global embree data */
struct EeveeEmbreeData {
  RTCDevice device; /* embree device */
  RTCScene scene;   /* embree scene */


  //RTCRay 

  // stats and capabilities
  bool NATIVE_RAY4_ON, NATIVE_RAY8_ON, NATIVE_RAY16_ON;
  bool RAY_STREAM_ON;
  uint TASKING_SYSTEM;
};


/* Functions */
void EVEM_init(void);
void EVEM_print_capabilities(void);
void EVEM_objects_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata);
void EVEM_objects_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool *cast_shadow);
void EVEM_free(void);

void EVEM_add_test_geo(void);
void EVEM_object_update_transform(Object *ob);
void EVEM_create_trimesh_geometry(Object *ob);

#ifdef  __cplusplus
}
#endif

#endif // __EEVEE_PRIVATE_H__