#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <search.h>

#include <omp.h>

#include "eevee_embree_objects.h"


/* embree scene */

ObjectsMap embree_objects_map = {.root=NULL, .items=NULL, .size=0, .alloc_size=0};

// obj map compare fuction that uses (ObjectsMapItem *)'s as both r and l arguments
int _evem_ob_map_compare(const void *l, const void *r) {
  const ObjectsMapItem *ll = (const ObjectsMapItem *)l;
  const ObjectsMapItem *rr = (const ObjectsMapItem *)r;
  
  if (ll->key.ob != rr->key.ob)
  	return ll->key.ob - rr->key.ob;

  return ll->key.ob->parent - rr->key.ob->parent;
};

// obj map compare fuction that uses (Object *) as l argument to compare
int _evem_ob_map_compare_ob(const void *l, const void *r) {
  const Object *ob = (const Object *)l;
  const ObjectsMapItem *rr = (const ObjectsMapItem *)r;
  
  if (ob != rr->key.ob)
  	return ob - rr->key.ob;

  return ob->parent - rr->key.ob->parent;
};


void _evem_ob_free_item(void *node) {
  free( (ObjectsMapItem *)node );
}

void _evem_ob_map_grow(void) {
	if(!embree_objects_map.items || (embree_objects_map.size < embree_objects_map.alloc_size))
		return;
	
	ObjectsMapItem **tmp = malloc(sizeof(ObjectsMapItem*) * embree_objects_map.size);
	memcpy(tmp, embree_objects_map.items, sizeof(ObjectsMapItem*) * embree_objects_map.size);

	free(embree_objects_map.items);
	embree_objects_map.items = malloc(sizeof(ObjectsMapItem*) * (embree_objects_map.size + EMEV_OBMAP_CHUNK_SIZE));
	memcpy(embree_objects_map.items, tmp, sizeof(ObjectsMapItem*) * embree_objects_map.size);
	free(tmp);

	embree_objects_map.alloc_size = embree_objects_map.size + EMEV_OBMAP_CHUNK_SIZE;
};


void EVEM_objects_map_init(void) {
	if(embree_objects_map.items && (embree_objects_map.alloc_size != 0))
		return;

	embree_objects_map.items = malloc(sizeof(ObjectsMapItem*) * EMEV_OBMAP_CHUNK_SIZE);
	embree_objects_map.root = NULL;
	embree_objects_map.size = 0;
	embree_objects_map.alloc_size = EMEV_OBMAP_CHUNK_SIZE;
};

void EVEM_objects_map_free(void) {
	if(!embree_objects_map.items)
		return;

	for (uint i = 0; i < (sizeof(embree_objects_map.items) / sizeof(ObjectsMapItem)); i++) {
		free(embree_objects_map.items[i]);
	}

	free(embree_objects_map.items);
	if (embree_objects_map.root) free(embree_objects_map.root); // TODO: tdestroy() should be here
	embree_objects_map.items = NULL;
	embree_objects_map.root = NULL;
	embree_objects_map.size = 0;
	embree_objects_map.alloc_size = 0;
};

//
// The function that inserts an item into the tree.
//
ObjectInfo *EVEM_insert_object(const Object *ob) {
	if(!ob) return NULL; // early termination
	if(EVEM_find_object_info(ob)) return NULL; // object already presented in map

	//printf("EVEM_insert_object: %s\n", ob->id.name);
	
	/* Create the (new) item */
  ObjectsMapItem *new_item = malloc(sizeof(ObjectsMapItem));
  new_item->key.ob = ob;
  new_item->info.ob = ob;
  new_item->info.id = 0;
  new_item->info.is_visible = false;
  new_item->info.escene = NULL;
  new_item->info.geometry = NULL;
  new_item->info.cast_shadow = true;
  new_item->info.deleted_or_hidden = false;

  ObjectsMapItem **item_in_tree = tsearch(new_item, &embree_objects_map.root, _evem_ob_map_compare);

  if (!item_in_tree) return NULL; // we shouldn't be here but ....
  
  // We've just inserted element, let's store it in items
  _evem_ob_map_grow();
  embree_objects_map.items[embree_objects_map.size++] = new_item;
  return &new_item->info; //&(*item_in_tree)->info;
}

//
// A function that finds an object info.
//
ObjectInfo *EVEM_find_object_info(const Object *ob) {
	if(!ob || (embree_objects_map.size == 0 )) return NULL; // early termination

  ObjectsMapItem **item_in_tree = tfind(ob, &embree_objects_map.root, _evem_ob_map_compare_ob);
  
  if (item_in_tree) {
    //printf("Found existing object %s info\n", ob->id.name);
    return &(*item_in_tree)->info;
  } 

  return NULL;
}
