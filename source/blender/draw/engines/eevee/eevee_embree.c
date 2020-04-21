#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "DNA_meshdata_types.h"
#include "BLI_math_geom.h"

#include "eevee_embree.h"


struct EeveeEmbreeData evem_data = {NULL};
static bool _scene_is_empty = true; // HACK for being able to test geometry already sent to embree device.

void EVEM_init(void) {
  if (!evem_data.device) {
    printf("%s\n", "create embree device");
    evem_data.device = rtcNewDevice(NULL);//rtcNewDevice("threads=0");
    assert(evem_data.device && "Unable to create embree device !!!");
  }

  if (!evem_data.scene) {
    printf("%s\n", "create embree scene");
    evem_data.scene = rtcNewScene(evem_data.device);
    assert(evem_data.scene);
    rtcSetSceneBuildQuality(evem_data.scene, RTC_BUILD_QUALITY_HIGH);
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
			EVEM_create_trimesh_geometry(ob);

		} else {
			/* instance */
			//RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
		}
		printf("%s %s\n", "MEEAASH!!!!", dupli_parent);
	}
	rtcCommitScene(evem_data.scene);
}

void EVEM_create_trimesh_geometry(Object *ob) {
	clock_t tstart = clock();
	uint ob_id = ob->id.session_uuid; // object uuid

	if(!_scene_is_empty) {
		if (rtcGetGeometry(evem_data.scene, ob_id)) {
			printf("%s\n", "Geometry already exist in Embree scene !");
			return;
		}
	}

	struct Mesh * mesh = (struct Mesh *)ob->data; // blender object mesh data
	//printf("%d polys\n", mesh->totpoly);

	uint vtc_count = mesh->totvert;
	uint tri_count = poly_to_tri_count(mesh->totpoly, mesh->totloop );
	//printf("%d tris\n", tri_count);

	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE); // embree geometry

	/* map triangle and vertex buffer */
  EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EVEM_Vertex3f), vtc_count);
  EVEM_Triangle* triangles = (EVEM_Triangle*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EVEM_Triangle), tri_count);

  // fill vertex buffer
  for (int i = 0; i < vtc_count; i++) {
  	vertices[i].x = mesh->mvert[i].co[0];
  	vertices[i].y = mesh->mvert[i].co[1];
  	vertices[i].z = mesh->mvert[i].co[2];
  }
	
  // fill triangles buffer
  int ti = 0; // triangles buffer index
  int cli = 0; // current mloop index

  MPoly *curr_mpoly;

  for (int i = 0; i < mesh->totpoly; i++) {
  	curr_mpoly = &mesh->mpoly[i];
  	cli = curr_mpoly->loopstart;

  	if (curr_mpoly->totloop == 3) {
  		// triangle ...
  		triangles[ti].v0 = mesh->mloop[cli++].v;
  		triangles[ti].v1 = mesh->mloop[cli++].v;
  		triangles[ti].v2 = mesh->mloop[cli++].v;
  	} else if(curr_mpoly->totloop > 3) {
  		// ngon ...
  	
  	}
  }

	// final steps 
	//rtcEnableGeometry(geometry);
	rtcCommitGeometry(geometry);
  rtcAttachGeometryByID(evem_data.scene, geometry, ob_id); // Attach geometry to Embree scene
  //rtcReleaseGeometry(geometry);

  _scene_is_empty = false;

  clock_t tend = clock();
  printf("Geoemtry synced in %f seconds\n", (double)(tend - tstart) / CLOCKS_PER_SEC);

}

// rtcCommitScene(RTCScene scene);