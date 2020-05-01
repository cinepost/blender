#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <search.h>

#include <omp.h>

#include "eevee_embree_objects.h"


/* embree scene */

static struct ObjectsMap embree_objects_map = {.root=NULL, .items=NULL, .size=0, .alloc_size=0};

/* objects map */
int _evem_ob_map_compare(const void *l, const void *r) {
  const ObjectsMapItem *ll = (const ObjectsMapItem *)l;
  const ObjectsMapItem *rr = (const ObjectsMapItem *)r;
  //return 0;
  return ll->key.ob - rr->key.ob;
};

void _evem_ob_free_item(void *node) {
  free( (ObjectsMapItem *)node );
}

void _evem_ob_map_grow(void) {
	if (embree_objects_map.size < embree_objects_map.alloc_size)
		return; // it's too early to grow 
	
	ObjectsMapItem *tmp = malloc(sizeof(embree_objects_map.items));
	memcpy(tmp, embree_objects_map.items, sizeof(embree_objects_map.items));

	free(embree_objects_map.items);
	embree_objects_map.items = malloc(sizeof(ObjectsMapItem*) * (embree_objects_map.size + EMEV_OBMAP_CHUNK_SIZE));
	memcpy(embree_objects_map.items, tmp, sizeof(tmp));
	free(tmp);

	embree_objects_map.alloc_size = embree_objects_map.size + EMEV_OBMAP_CHUNK_SIZE;
};

void EVEM_objects_map_init(void) {
	if(embree_objects_map.items && (embree_objects_map.alloc_size != 0))
		return;

	embree_objects_map.items = malloc(sizeof(ObjectsMapItem*) * EMEV_OBMAP_CHUNK_SIZE);
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
	embree_objects_map.size = 0;
	embree_objects_map.alloc_size = 0;
};

//
// The function that inserts an item into the tree.
//
ObjectInfo *EVEM_insert_object(const Object *ob) {
	printf("EVEM_insert_object: %s\n", ob->id.name);
	//
	// Create the (new) item
	//
  ObjectsMapItem *new_item = malloc(sizeof(ObjectsMapItem));
  new_item->key.ob = ob;
  new_item->info.ob = ob;
  new_item->info.is_visible = false;

	//
	// tsearch searches for a node with a specific key.
	// If no such node was found, it inserts it and returns it.
	//
  ObjectsMapItem **item_in_tree = tsearch(new_item, &embree_objects_map.root, _evem_ob_map_compare);

  if (*item_in_tree != new_item) {
	  printf("Object already inserted: %s, overriting\n", ob->id.name);

    // Because the key was already inserted, we overwrite the
    // already inserted item's value with the new value
    (*item_in_tree)->info = new_item->info;

    // The new item is not needed anymore:
    free(new_item);
  }

  // We've just inserted element, let's store it in items
  _evem_ob_map_grow();
  embree_objects_map.items[embree_objects_map.size++] = new_item;

  return &(*item_in_tree)->info;
}

//
// A function that finds an object info.
//
ObjectInfo *EVEM_find_object_info(const Object *ob) {
  ObjectsMapItem *tmp_item = malloc(sizeof(ObjectsMapItem));
  tmp_item->key.ob = ob;

  ObjectsMapItem **item_in_tree = tfind(tmp_item, &embree_objects_map.root, _evem_ob_map_compare);
  free(tmp_item);

  if (item_in_tree) {
    printf("Found existing object %s info\n", ob->id.name);
    return &(*item_in_tree)->info;
  } 

  return NULL;
}
