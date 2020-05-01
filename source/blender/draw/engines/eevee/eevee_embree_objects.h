#ifndef __EEVEE_EMBREE_OBJECTS_H__
#define __EEVEE_EMBREE_OBJECTS_H__

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

#define EMEV_OBMAP_CHUNK_SIZE 16 // objects map preallocation chunk size

enum { OBJECT_PERSISTENT_ID_SIZE = 16 };

typedef struct {
  void *ob;
  void *parent;
  int id[OBJECT_PERSISTENT_ID_SIZE];
  bool use_particle_hair;
} ObjectKey;

typedef struct {
  Object *ob;
	uint id; // embree geometry id
	bool is_visible;
} ObjectInfo;

typedef struct {
  ObjectKey   key;
  ObjectInfo  info; 
} ObjectsMapItem;

struct ObjectsMap {
  void            *root;
  ObjectsMapItem  **items;
  uint size;        // used items size
  uint alloc_size;  // allocated size in items count
};

/* "private" functions */
int   _evem_ob_map_compare(const void *l, const void *r);
void _evem_ob_free_item(void *node);

/* Functions */
void  EVEM_objects_map_init(void);
void  EVEM_objects_map_free(void);
ObjectInfo *EVEM_insert_object(const Object *ob);
ObjectInfo *EVEM_find_object_info(const Object *ob);
bool EVEM_object_exist(const Object *ob);


#ifdef  __cplusplus
}
#endif

#endif // __EEVEE_EMBREE_OBJECTS_H__