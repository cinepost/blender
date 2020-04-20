#include <stdio.h>
#include <assert.h>

#include "DNA_meshdata_types.h"
#include "BLI_math_geom.h"

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
			EVEM_create_object(ob);

		} else {
			/* instance */

		}
		printf("%s %s\n", "MEEAASH!!!!", dupli_parent);
	}

}

void EVEM_create_object(Object *ob) {
	uint ob_id = ob->id.session_uuid; // object uuid
	struct Mesh * mesh = (struct Mesh *)ob->data; // blender object mesh data
	printf("%d polys\n", mesh->totpoly);

	uint vtc_count = mesh->totvert;
	uint tri_count = poly_to_tri_count(mesh->totpoly, mesh->totloop );
	printf("%d tris\n", tri_count);
	//MPoly *mpoly;
	//for(int i=0; i < mesh->totpoly; i++ ) {
	//	mpoly = &mesh->mpoly[i];
	//	printf("%d poly loops\n", mpoly->totloop);
	//}

	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE); // embree geometry

	/* map triangle and vertex buffer */
  EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EVEM_Vertex3f), vtc_count);
  EVEM_Triangle* triangles = (EVEM_Triangle*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EVEM_Triangle), tri_count);
	
	// final steps 
	rtcCommitGeometry(geometry);
  rtcAttachGeometryByID(evem_data.scene, geometry, ob_id); // Attach geometry to Embree scene
  rtcReleaseGeometry(geometry);
}

// rtcCommitScene(RTCScene scene);