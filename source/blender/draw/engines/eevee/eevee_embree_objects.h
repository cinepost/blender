#ifndef __EEVEE_EMBREE_OBJECTS_H__
#define __EEVEE_EMBREE_OBJECTS_H__

#include <embree3/rtcore.h>
#include <embree3/rtcore_scene.h>

#include "BKE_object.h"
#include "DEG_depsgraph_query.h"
#include "DNA_world_types.h"
#include "DNA_mesh_types.h"

#include "eevee_private.h"

#ifdef _WIN32
#include "tq84-tsearch.h"
#define tsearch(X, Y, Z) tq84_tsearch(X, Y, Z)
#define tfind(X, Y, Z) tq84_tfind(X, Y, Z)
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define EMEV_OBMAP_CHUNK_SIZE 16 // objects map preallocation chunk size

enum { OBJECT_PERSISTENT_ID_SIZE = 16 };

typedef struct {
  Object *ob;
  void *parent;
  int id[OBJECT_PERSISTENT_ID_SIZE];
  bool use_particle_hair;
} ObjectKey;

typedef struct {
  Object *ob;
	uint id; // embree geometry id
  RTCGeometry geometry;
  bool is_rtc_instance;
	bool is_visible;
  bool cast_shadow;
  bool deleted_or_hidden;
  RTCScene escene;
  float xform[16];
} ObjectInfo;

typedef struct {
  ObjectKey   key;
  ObjectInfo  info; 
} ObjectsMapItem;

typedef struct {
  void           *root;
  ObjectsMapItem **items;
  uint size;       // used items size
  uint alloc_size; // allocated size in items count
} ObjectsMap;

/* "private" functions */
int _evem_ob_map_compare(const void *l, const void *r);
int _evem_ob_map_compare_ob(const void *l, const void *r);
void _evem_ob_free_item(void *node);

/* Functions */
void  EVEM_objects_map_init(void);
void  EVEM_objects_map_free(void);
ObjectInfo *EVEM_insert_object(const Object *ob);
ObjectInfo *EVEM_find_object_info(const Object *ob);
bool EVEM_object_exist(const Object *ob);

extern ObjectsMap embree_objects_map;

#ifdef  __cplusplus
}
#endif

#endif // __EEVEE_EMBREE_OBJECTS_H__