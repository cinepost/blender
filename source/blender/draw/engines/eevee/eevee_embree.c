#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "DNA_meshdata_types.h"
#include "BLI_math_geom.h"

#include "eevee_embree.h"


struct EeveeEmbreeData evem_data = {NULL};
static bool _scene_is_empty = true; // HACK for being able to test geometry already sent to embree device.
static bool _evem_inited = false;

void EVEM_init(void) {
	if (_evem_inited) return;

  if (!evem_data.device) {
    printf("%s\n", "create embree device");
    evem_data.device = rtcNewDevice("threads=6");
    assert(evem_data.device && "Unable to create embree device !!!");
  }

	evem_data.NATIVE_RAY4_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);
	evem_data.NATIVE_RAY8_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
  evem_data.NATIVE_RAY16_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
  evem_data.RAY_STREAM_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_RAY_STREAM_SUPPORTED);
  evem_data.TASKING_SYSTEM = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_TASKING_SYSTEM);

  if (!evem_data.scene) {
    printf("%s\n", "create embree scene");
    evem_data.scene = rtcNewScene(evem_data.device);
    assert(evem_data.scene);
    rtcSetSceneFlags(evem_data.scene, RTC_SCENE_FLAG_DYNAMIC);
    rtcSetSceneBuildQuality(evem_data.scene, RTC_BUILD_QUALITY_HIGH);
  }

  _evem_inited = true;
  EVEM_print_capabilities();
}

void EVEM_print_capabilities(void) {
	if (!_evem_inited) return;
	printf("Embree3 capabilities...\n_________________________\n");
	printf("Embree3 native Ray4 %s\n", evem_data.NATIVE_RAY4_ON ? "ON":"OFF");
	printf("Embree3 native Ray8 %s\n", evem_data.NATIVE_RAY8_ON ? "ON":"OFF");
	printf("Embree3 native Ray16 %s\n", evem_data.NATIVE_RAY16_ON ? "ON":"OFF");
	printf("Embree3 Ray stream %s\n", evem_data.RAY_STREAM_ON ? "ON":"OFF");

	printf("Embree3 Tasking system: ");
	switch(evem_data.TASKING_SYSTEM) {
		case(0):
			printf("internal\n");
			break;
		case(1):
			printf("TBB\n");
			break;
		case(2):
			printf("PLL\n");
			break;

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

void EVEM_add_test_geo(void) {
	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE);

	EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EVEM_Vertex3f), 4);
  EVEM_Triangle* triangles = (EVEM_Triangle*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EVEM_Triangle), 2);

  vertices[0].x = 0.0; vertices[1].x = 0.0; vertices[2].x = 0.0; vertices[3].x = 0.0;
  vertices[0].y = 0.0; vertices[1].y = 1.0; vertices[2].y = 1.0; vertices[3].y = 0.0;
  vertices[0].z = 0.0; vertices[1].z = 0.0; vertices[2].z = 1.0; vertices[3].z = 1.0;

  triangles[0].v0 = 0; triangles[0].v1 = 1; triangles[0].v2 = 2;
  triangles[1].v0 = 0; triangles[1].v1 = 2; triangles[1].v2 = 3;

  rtcCommitGeometry(geometry);
  rtcAttachGeometry(evem_data.scene, geometry);
 }

void EVEM_create_trimesh_geometry(Object *ob) {
	clock_t tstart = clock();
	uint ob_id = ob->id.session_uuid; // object uuid

	if(!_scene_is_empty) {
		if (rtcGetGeometry(evem_data.scene, ob_id)) {
			EVEM_object_update_transform(ob); // update object transform
			printf("%s\n", "Geometry already exist in Embree scene !");
			return;
		}
	}

	struct Mesh * mesh = (struct Mesh *)ob->data; // blender object mesh data
	//printf("%d polys\n", mesh->totpoly);

	uint vtc_count = mesh->totvert;
	uint tri_count = poly_to_tri_count(mesh->totpoly, mesh->totloop );
	printf("%d tris\n", tri_count);

	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE); // embree geometry
	rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
	//rtcSetGeometryTimeStepCount(geometry, 0);
	rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, (const float *)&ob->obmat); // set transform

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
  
  MPoly *curr_mpoly;

  for (int i = 0; i < mesh->totpoly; i++) {
  	curr_mpoly = &mesh->mpoly[i];
  	
  	if (curr_mpoly->totloop == 3) {
  		// triangle ...
  		triangles[ti].v0 = mesh->mloop[curr_mpoly->loopstart].v;
  		triangles[ti].v1 = mesh->mloop[curr_mpoly->loopstart+1].v;
  		triangles[ti].v2 = mesh->mloop[curr_mpoly->loopstart+2].v;
  		ti++;
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

void EVEM_object_update_transform(Object *ob) {
	printf("%s\n", "embree update object");

	if(_scene_is_empty)return;

	uint ob_id = ob->id.session_uuid; // object uuid
	RTCGeometry geometry = rtcGetGeometry(evem_data.scene, ob_id);

	if (!geometry) {
		printf("%s\n", "Geometry already exist in Embree scene !");
		return;
	}

	//float xfm[16] = {1.0f};
  //xfm[0] = ob->loc[0];
  //xfm[1] = ob->loc[1];
  //xfm[2] = ob->loc[2];

  //xfm[4] = ob->scale[0];
  //xfm[5] = ob->scale[1];
  //xfm[6] = ob->scale[2];

  //printf("object location %f %f %f\n", ob->loc[0], ob->loc[1], ob->loc[2]);

  //rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, xfm);
  rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, (const float *)&ob->obmat); // set transform
  //rtcCommitGeometry(geometry);
  rtcCommitScene(evem_data.scene);

  printf("%s\n", "embree object updated");
}

// rtcCommitScene(RTCScene scene);