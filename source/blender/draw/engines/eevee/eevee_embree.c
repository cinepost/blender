#include <stdio.h>
#include <assert.h>

#include "eevee_embree.h"


struct EeveeEmbreeData evem_data = {NULL};

void EVEM_init(void) {
  if (!evem_data.device) {
    printf("%s\n", "create embree device");
    evem_data.device = rtcNewDevice("threads=0");
    assert(evem_data.device && "Unable to create embree device !!!");
  }

  if (!evem_data.scene) {
    printf("%s\n", "create embree scene");
    evem_data.scene = rtcNewScene(evem_data.device);
    assert(scene);
  }
}

void EVEM_free(void) {
	rtcReleaseScene(evem_data.scene);
	rtcReleaseDevice(evem_data.device);
}

void EVEM_objects_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata) {
	printf("%s\n", "EVEM_objects_cache_init");
}

void EVEM_objects_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool *cast_shadow) {
	if (!cast_shadow)
		return;

	printf("%s\n", "EVEM_objects_cache_populate");
	Object *dupli_parent;
	if (ELEM(ob->type, OB_MESH)) {
		dupli_parent = DRW_object_get_dupli_parent(ob);
		if (!dupli_parent){
			/* object itself */
			struct Mesh * mesh = (struct Mesh *)ob->data;
			printf("%d vertices\n", mesh->totvert);
		} else {
			/* instance */

		}
		printf("%s %s\n", "MEEAASH!!!!", dupli_parent);
	}

}