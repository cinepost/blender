#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <search.h>

#include <omp.h>

#include "DNA_scene_types.h"
#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"
#include "BLI_math_geom.h"

#include "BKE_editmesh.h"

#include "GPU_batch.h"

#include "draw_cache_extract.h"
#include "draw_cache_impl.h"

#include <xmmintrin.h>
#include <pmmintrin.h>

#include "eevee_embree.h"

/* embree scene */

struct EeveeEmbreeData evem_data = {NULL, .update_tlas = false, .update_blas = false, .sample_num = 1.0f, .embree_enabled=false, .image_render_mode=false};
static bool _scene_is_empty = true; // HACK for being able to test geometry already sent to embree device.
static bool _evem_inited = false;

// USE_FLAT_SCENE means no instancing. all the geometries goes into evem_data.scene directtly
#define USE_FLAT_SCENE true

void EEVEE_embree_init(EEVEE_ViewLayerData *sldata) {
  if( _evem_inited ) return;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  if (!scene_eval->eevee.flag & SCE_EEVEE_RTAO_ENABLED) {
    evem_data.embree_enabled = false;
    return;
  }

  evem_data.embree_enabled = true;

  if (!evem_data.device) {
    printf("%s\n", "create embree device");
    evem_data.device = rtcNewDevice("threads=0");//,verbose=3");
    assert(evem_data.device && "Unable to create embree device !!!");
  }

  if(!evem_data.scene) { // rtcReleaseScene(evem_data.scene);
    evem_data.scene = rtcNewScene(evem_data.device);
    rtcSetSceneFlags(evem_data.scene, RTC_SCENE_FLAG_ROBUST);// | RTC_SCENE_FLAG_DYNAMIC);
    rtcSetSceneBuildQuality(evem_data.scene, RTC_BUILD_QUALITY_HIGH);
  }

	evem_data.NATIVE_RAY4_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED);
	evem_data.NATIVE_RAY8_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED);
  evem_data.NATIVE_RAY16_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED);
  evem_data.RAY_STREAM_ON = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_RAY_STREAM_SUPPORTED);
  evem_data.TASKING_SYSTEM = rtcGetDeviceProperty(evem_data.device, RTC_DEVICE_PROPERTY_TASKING_SYSTEM);

  EEVEE_embree_print_capabilities();

  EVEM_objects_map_init();

  _evem_inited = true;

}

void EEVEE_embree_print_capabilities(void) {
	if (_evem_inited) return;
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

void EEVEE_embree_free(void) {
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

void EEVEE_embree_cache_init(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata) {
  if (!evem_data.embree_enabled || (evem_data.sample_num > 1.0f)) return; // init only on first sample

  printf("EEVEE_embree_cache_init\n");

  bool image_render_mode = DRW_state_is_image_render() ? true : false;

  if (evem_data.image_render_mode != image_render_mode ) {
    evem_data.image_render_mode = image_render_mode;
    /* If mode changed from viewport to image or other way it
    /* ooks like we need to rebuild obj map and clear scene */
    EVEM_objects_map_free();
    EVEM_objects_map_init();
    rtcReleaseScene(evem_data.scene);
    evem_data.scene = rtcNewScene(evem_data.device);
    rtcSetSceneFlags(evem_data.scene, RTC_SCENE_FLAG_ROBUST);// | RTC_SCENE_FLAG_DYNAMIC);
    rtcSetSceneBuildQuality(evem_data.scene, RTC_BUILD_QUALITY_HIGH);
  }


  ObjectInfo *ob_info = NULL;
  
  for(uint i=0; i < embree_objects_map.size; i++) {
    ob_info = &embree_objects_map.items[i]->info;

    // hide all objects before populating cache amd mark them as candidates for later deletion 
    if(ob_info->geometry) {
      ob_info->delete_later = true;
      rtcDisableGeometry(ob_info->geometry);
    }
  }
}

void EEVEE_embree_cache_populate(EEVEE_Data *vedata, EEVEE_ViewLayerData *sldata, Object *ob, bool cast_shadow) {
  if (!evem_data.embree_enabled || (evem_data.sample_num > 1.0f)) return; // updates only on first sample

  EEVEE_StorageList *stl = vedata->stl;
  EEVEE_EffectsInfo *effects = stl->effects;

  const DRWContextState *draw_ctx = DRW_context_state_get();
  const Scene *scene_eval = DEG_get_evaluated_scene(draw_ctx->depsgraph);

  if (!(scene_eval->eevee.flag & SCE_EEVEE_RTAO_ENABLED)) return;

  int sample = (DRW_state_is_image_render()) ? effects->taa_render_sample : effects->taa_current_sample;
  if (sample > 1) return; // create/update only on first sample

  printf("EVEM_objects_cache_populate\n");
  
	//if (!cast_shadow)
	//	return;

  ObjectInfo *ob_info = EVEM_find_object_info(ob);
  if(!ob_info) {
    ob_info = EVEM_insert_object(ob);
    if (!ob_info)
      return; // trying to insert object into unitialized map

    EVEM_create_object(ob, ob_info);
  } else {
    EVEM_update_object(ob, ob_info);
  }

  /* make object visible */
  if(ob_info->geometry)
    rtcEnableGeometry(ob_info->geometry);

  ob_info->delete_later = false; // we've got the object.. we good
  evem_data.update_tlas = true;
}

/* here we acutally upate tlas */
void EEVEE_embree_cache_finish(EEVEE_ViewLayerData *sldata, EEVEE_Data *vedata) {
  if(!evem_data.embree_enabled || !evem_data.update_tlas || (!evem_data.sample_num > 1.0f)) return;
  printf("EVEM_objects_cache_finish (update TLAS)\n");

  /* check and possibly delete candidates */ 
  ObjectInfo *ob_info = NULL;
  
  for(uint i=0; i < embree_objects_map.size; i++) {
    ob_info = &embree_objects_map.items[i]->info;
    if (ob_info->delete_later && !ob_info->ob->data) {
      // delete candidate that has no data (ob->data == NULL) 
      printf("Delete object: \n", ob_info->ob->id.name);
    }
  }

  rtcCommitScene(evem_data.scene);
  _scene_is_empty = false;
  evem_data.update_tlas = false; // tlas already updated
}

void EVEM_create_object(Object *ob, ObjectInfo *ob_info) {
	//printf("EVEM_create_object: %s\n", ob->id.name);

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
  //printf("EVEM_update_object: %s\n", ob->id.name);

  switch (ob->type) {
    case OB_MESH:
      if(USE_FLAT_SCENE) {
        EVEM_mesh_object_update(ob, ob_info);
      } else {
        EVEM_instance_update_transform(ob, ob_info);
      }
      break;
    default:
      break;
  }
}


void EVEM_mesh_object_clear(Mesh *me) {

}

void EVEM_mesh_object_create(Mesh *me, ObjectInfo *ob_info) {
  printf("EVEM_mesh_object_create for ob_info->id: %u \n", ob_info->id);
	clock_t tstart = clock();

  if (ob_info->geometry) {
    printf("Warning! Embree geometry already created for Mesh object: %s\n", ob_info->ob->id.name);
    return;
  }

  uint vtc_count = me->totvert;
  uint tri_count = poly_to_tri_count(me->totpoly, me->totloop );

  /* unsupported mesh */
  if((tri_count < 1) || (vtc_count < 3)) return;


  if(!USE_FLAT_SCENE) {
	  // geometry goes to local scene
    if (ob_info->escene) rtcReleaseScene(ob_info->escene);
    ob_info->escene = rtcNewScene(evem_data.device);
    rtcSetSceneFlags(ob_info->escene, RTC_SCENE_FLAG_ROBUST);// | RTC_SCENE_FLAG_DYNAMIC);
    rtcSetSceneBuildQuality(ob_info->escene, RTC_BUILD_QUALITY_HIGH);
  }

	RTCGeometry geometry = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_TRIANGLE); // embree geometry
	rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);

  if(!USE_FLAT_SCENE) {
    rtcSetGeometryTimeStepCount(geometry,1);
    rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (const float *)&ob_info->ob->obmat[0]);
  }

	/* map triangle and vertex buffer */
  
  EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(EVEM_Vertex3f), vtc_count);
  EVEM_Triangle* triangles = (EVEM_Triangle*) rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(EVEM_Triangle), tri_count);

  // fill vertex buffer
  if(!USE_FLAT_SCENE) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < vtc_count; i++) {
  	 vertices[i].x = me->mvert[i].co[0];
  	 vertices[i].y = me->mvert[i].co[1];
  	 vertices[i].z = me->mvert[i].co[2];
    }
	} else {
    // pre transformed vertices for flat scene
    float *m = (const float *)&ob_info->ob->obmat[0];
    #pragma omp parallel for num_threads(8) shared(m)
    for (int i = 0; i < vtc_count; i++) {
      vertices[i].x = m[0] * me->mvert[i].co[0] + m[4] * me->mvert[i].co[1] + m[8] * me->mvert[i].co[2] + m[12];
      vertices[i].y = m[1] * me->mvert[i].co[0] + m[5] * me->mvert[i].co[1] + m[9] * me->mvert[i].co[2] + m[13];
      vertices[i].z = m[2] * me->mvert[i].co[0] + m[6] * me->mvert[i].co[1] + m[10] * me->mvert[i].co[2] + m[14];
    }
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
      case 5:
        // pentagon
        ti+=3;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+1].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+2].v;
        curr_idx++;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+2].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+3].v;
        curr_idx++;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+3].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+4].v;
        break;
      case 6:
        // hexagon
        ti+=4;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+1].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+2].v;
        curr_idx++;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+2].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+3].v;
        curr_idx++;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+3].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+5].v;
        curr_idx++;
        triangles[curr_idx].v0 = me->mloop[curr_mpoly->loopstart+5].v;
        triangles[curr_idx].v1 = me->mloop[curr_mpoly->loopstart+3].v;
        triangles[curr_idx].v2 = me->mloop[curr_mpoly->loopstart+4].v;
        break;
  		default:
  			// ngon
        ti+= curr_mpoly->totloop - 2;
        for(uint i=0; i < (curr_mpoly->totloop - 2); i++) {
          triangles[curr_idx+i].v0 = me->mloop[curr_mpoly->loopstart].v;
          triangles[curr_idx+i].v1 = me->mloop[curr_mpoly->loopstart+i+1].v;
          triangles[curr_idx+i].v2 = me->mloop[curr_mpoly->loopstart+i+2].v;
        }
  			break; 
  	}
  }

	rtcCommitGeometry(geometry);

  if(!USE_FLAT_SCENE) {
    rtcAttachGeometryByID(ob_info->escene, geometry, 0); // local object geometry to local scene scene id
    rtcCommitScene(ob_info->escene);

    RTCGeometry inst = rtcNewGeometry(evem_data.device, RTC_GEOMETRY_TYPE_INSTANCE);
    rtcSetGeometryBuildQuality(inst, RTC_BUILD_QUALITY_HIGH);
    rtcSetGeometryInstancedScene(inst, ob_info->escene);
    rtcSetGeometryTimeStepCount(inst,1);
    rtcSetGeometryTransform(inst, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (const float *)&ob_info->ob->obmat[0]);
    rtcCommitGeometry(inst);
    
    ob_info->id = rtcAttachGeometry(evem_data.scene, inst);
    ob_info->geometry = inst;
    rtcReleaseGeometry(inst);
  } else {
    ob_info->id = rtcAttachGeometry(evem_data.scene, geometry);
    ob_info->geometry = geometry;
    memcpy(ob_info->xform, &ob_info->ob->obmat[0], sizeof(float)*16); 
  }

  rtcReleaseGeometry(geometry);

  evem_data.update_tlas = true;

  clock_t tend = clock();
  //printf("Mesh geometry for object %s with embree id %u added in %f seconds\n",  ob_info->ob->id.name, ob_info->id, (double)(tend - tstart) / CLOCKS_PER_SEC);
}

void EVEM_mesh_object_update(Object *ob, ObjectInfo *ob_info) {
  if(_scene_is_empty)return;
	//printf("%s\n", "EVEM_mesh_object_update");

  // return if object transform not changed and object not in edit mode
  bool is_edit_mode = false;
  if(!memcmp((const void *)&ob_info->xform[0], (const void *)&ob->obmat[0], sizeof(float)*16)) {
    // object thansform is the same. check if we in edit mode
    is_edit_mode = DRW_object_is_in_edit_mode(ob);
    if (!is_edit_mode) return; // same transform, not in edit mode. return
  }

	RTCGeometry geometry = rtcGetGeometry(evem_data.scene, ob_info->id);
  
  if (!geometry) {
    printf("%s\n", "Error can't update missing Embree geometry !");
    return;
  }

  EVEM_Vertex3f* vertices  = (EVEM_Vertex3f*) rtcGetGeometryBufferData(geometry, RTC_BUFFER_TYPE_VERTEX, 0);

  struct Mesh *me = NULL;

  if (is_edit_mode) {
    struct Mesh *_me = (Mesh *)ob->data;
    struct BMEditMesh *embm = _me->edit_mesh;
    if (embm) me = (Mesh *)embm->mesh_eval_final;
  } else {
    me = (Mesh *)ob->data;
  }

  if(!me) return;

  /* Check if the object that we are drawing is modified. */
  if (!DEG_is_original_id(&me->id)) {
    // TODO: resync mesh topology !
    return false;
  }

  float *m = (const float *)&ob_info->ob->obmat[0];
  #pragma omp parallel for shared(m)
  // embree wants our geometry to be pre transformed
  for (int i = 0; i < me->totvert; i++) {
    vertices[i].x = m[0] * me->mvert[i].co[0] + m[4] * me->mvert[i].co[1] + m[8] * me->mvert[i].co[2] + m[12];
    vertices[i].y = m[1] * me->mvert[i].co[0] + m[5] * me->mvert[i].co[1] + m[9] * me->mvert[i].co[2] + m[13];
    vertices[i].z = m[2] * me->mvert[i].co[0] + m[6] * me->mvert[i].co[1] + m[10] * me->mvert[i].co[2] + m[14];
  }

  /* commit mesh */
  rtcUpdateGeometryBuffer(geometry,RTC_BUFFER_TYPE_VERTEX,0);
  rtcCommitGeometry(geometry);
  
  evem_data.update_tlas = true;

  printf("%s\n", "embree object updated");
}

void EVEM_instance_update_transform(Object *ob, ObjectInfo *ob_info) {
  if(_scene_is_empty)return;
  printf("%s\n", "EVEM_instance_update_transform");

  RTCGeometry geometry = rtcGetGeometry(evem_data.scene, ob_info->id);
  if(!geometry) return;

  rtcSetGeometryTimeStepCount(geometry, 1);
  rtcSetGeometryTransform(geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (const float *)&ob->obmat[0]);
  rtcCommitGeometry(geometry);

  evem_data.update_tlas = true;
}

void EVEM_toggle_object_visibility(Object *ob, ObjectInfo *ob_info) {
  printf("EVEM_toggle_object_visibility\n");
  RTCGeometry geometry = ob_info->geometry;

  ob_info->cast_shadow = !ob_info->cast_shadow;

  if(ob_info->cast_shadow) {
    rtcEnableGeometry(geometry);
  } else {
    rtcDisableGeometry(geometry);
  }
}