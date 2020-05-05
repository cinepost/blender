#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <search.h>

#include <omp.h>

#include "DNA_scene_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "BLI_math_geom.h"

#include "GPU_batch.h"

#include "draw_cache_extract.h"
#include "draw_cache_impl.h"

#include "eevee_embree.h"

/* embree scene */

struct EeveeEmbreeData evem_data = {NULL};
static bool _scene_is_empty = true; // HACK for being able to test geometry already sent to embree device.
static bool _evem_inited = false;

void EVEM_init(void) {
	if (_evem_inited) return;

  if (!evem_data.device) {
    printf("%s\n", "create embree device");
    evem_data.device = rtcNewDevice("threads=8");
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
    rtcSetSceneFlags(evem_data.scene, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC);
    rtcSetSceneBuildQuality(evem_data.scene, RTC_BUILD_QUALITY_HIGH);
  }

  _evem_inited = true;
  EVEM_print_capabilities();

  EVEM_objects_map_init();
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
  EVEM_objects_map_free();
	rtcReleaseScene(evem_data.scene);
	rtcReleaseDevice(evem_data.device);
}

void EVEM_rays_buffer_free(struct EeveeEmbreeRaysBuffer *buff){
	if(!buff->rays && !buff->rays4 && !buff->rays8 && !buff->rays16) return;
  if(buff->rays16)free(buff->rays16);
  if(buff->rays8)free(buff->rays8);
  if(buff->rays4)free(buff->rays4);
  if(buff->rays)free(buff->rays);
}

void EVEM_objects_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata) {
	printf("EVEM_objects_cache_init\n");
}

void EVEM_objects_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool *cast_shadow) {
  printf("EVEM_objects_cache_populate\n");

	if (!cast_shadow)
		return;

  ObjectInfo *ob_info = EVEM_find_object_info(ob);
  if(!ob_info) {
    ob_info = EVEM_insert_object(ob);
    if (!ob_info)
      return; // trying to insert object into unitialized map

    EVEM_create_object(ob, ob_info);
  } else {
    EVEM_update_object(ob, ob_info);
  }
	rtcCommitScene(evem_data.scene);
}

void EVEM_create_object(Object *ob, ObjectInfo *ob_info) {
	printf("EVEM_create_object: %s\n", ob->id.name);
  struct Mesh *mesh_eval = NULL;
	switch (ob->type) {
    case OB_MESH:
      EVEM_mesh_object_create((Mesh *)ob->data, ob_info);
      break;
    case OB_CURVE:
    case OB_FONT:
    case OB_SURF:
      mesh_eval = BKE_object_get_evaluated_mesh(ob);
      if (mesh_eval != NULL) {
        EVEM_mesh_object_create(mesh_eval, ob_info);
      }
      //EVEM_curve_cache_validate((Curve *)ob->data);
      break;
    case OB_MBALL:
      //EVEM_mball_cache_validate((MetaBall *)ob->data);
      break;
    case OB_LATTICE:
      //EVEM_lattice_cache_validate((Lattice *)ob->data);
      break;
    case OB_HAIR:
      //EVEM_hair_cache_validate((Hair *)ob->data);
      break;
    default:
      break;
	}
}

void EVEM_update_object(Object *ob, ObjectInfo *ob_info) {

}


void EVEM_mesh_object_clear(Mesh *me) {

}

void EVEM_mesh_object_create(Mesh *me, ObjectInfo *ob_info) {
  printf("EVEM_mesh_object_create\n");
	clock_t tstart = clock();


	//if(!_scene_is_empty) {
	//	if (rtcGetGeometry(evem_data.scene, ob_info->id)) {
	//		printf("Mesh geometry with embree id %u for object %s already exist in Embree scene !\n", ob_info->id, ob_info->ob->id.name);
	//		return;
	//	}
	//}


	uint vtc_count = me->totvert;
	uint tri_count = poly_to_tri_count(me->totpoly, me->totloop );

	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE); // embree geometry
	rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
	//rtcSetGeometryTimeStepCount(geometry, 0);
	//rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, (const float *)&ob->obmat); // set transform

	/* map triangle and vertex buffer */
  EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EVEM_Vertex3f), vtc_count);
  EVEM_Triangle* triangles = (EVEM_Triangle*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EVEM_Triangle), tri_count);

  // fill vertex buffer
  #pragma omp parallel for num_threads(8)
  for (int i = 0; i < vtc_count; i++) {
  	vertices[i].x = me->mvert[i].co[0];
  	vertices[i].y = me->mvert[i].co[1];
  	vertices[i].z = me->mvert[i].co[2];
  }
	
  // fill triangles buffer
  int ti = 0; // triangles buffer index
  
  MPoly *curr_mpoly;

  //#pragma omp parallel for num_threads(8) private(curr_mpoly)  reduction(+:ti)
  for (int i = 0; i < me->totpoly; i++) {
  	uint curr_idx = ti;
  	curr_mpoly = &me->mpoly[i];
  	
  	switch(curr_mpoly->totloop) {
  		case 3:
  			// triangle
  			ti+=1;
  			triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
  			triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+1].v;
  			triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+2].v;
  			break;
  		case 4:
  			// quad
  		  ti+=2;
  			triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
  			triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+1].v;
  			triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+2].v;
				curr_idx++;
				triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
  			triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+2].v;
  			triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+3].v;
  			break;
  		default:
  			// ngon
  			break; 
  	}
  }

	// final steps 
	//rtcEnableGeometry(geometry);
	rtcCommitGeometry(geometry);
  ob_info->id = rtcAttachGeometry(evem_data.scene, geometry); // Attach geometry to Embree scene
  //rtcReleaseGeometry(geometry);

  _scene_is_empty = false;

  clock_t tend = clock();
  printf("Mesh geometry for object %s with embree id %u added in %f seconds\n",  ob_info->ob->id.name, ob_info->id, (double)(tend - tstart) / CLOCKS_PER_SEC);
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

//inline EVEM_Vec3f EVEM_mult_dir_matrix(EVEM_Matrix44f mat, EVEM_Vec3f vec) {
//	return EVEM_Vec3f(
//  	vec[0] * mat[0][0] + vec[1] * mat[1][0] + vec[2] * mat[2][0], 
//  	vec[0] * mat[0][1] + vec[1] * mat[1][1] + vec[2] * mat[2][1], 
//  	vec[0] * mat[0][2] + vec[1] * mat[1][2] + vec[2] * mat[2][2] 
//  );
//}

// rtcCommitScene(RTCScene scene);