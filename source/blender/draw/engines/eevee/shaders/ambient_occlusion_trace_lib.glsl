
/* Based on Practical Realtime Strategies for Accurate Indirect Occlusion
 * http://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf
 * http://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pptx
 */

#if defined(MESH_SHADER)
#  if !defined(USE_ALPHA_HASH)
#    if !defined(USE_ALPHA_CLIP)
#      if !defined(SHADOW_SHADER)
#        if !defined(USE_MULTIPLY)
#          if !defined(USE_ALPHA_BLEND)
#            define ENABLE_DEFERED_AO
#          endif
#        endif
#      endif
#    endif
#  endif
#endif

#ifndef ENABLE_DEFERED_AO
#  if defined(STEP_RESOLVE)
#    define ENABLE_DEFERED_AO
#  endif
#endif

#define MAX_PHI_STEP 32
#define MAX_SEARCH_ITER 32
#define MAX_LOD 6.0

#ifndef UTIL_TEX
#  define UTIL_TEX
uniform sampler2DArray utilTex;
#  define texelfetch_noise_tex(coord) texelFetch(utilTex, ivec3(ivec2(coord) % LUT_SIZE, 2.0), 0)
#endif /* UTIL_TEX */

uniform sampler2D ao_traceBuffer;

/* aoSettings flags */
#define USE_AO 1
#define USE_BENT_NORMAL 2
#define USE_DENOISE 4
#define USE_AO_TRACE 8

vec4 pack_horizons(vec4 v)
{
  return v * 0.5 + 0.5;
}
vec4 unpack_horizons(vec4 v)
{
  return v * 2.0 - 1.0;
}

/* Returns maximum screen distance an AO ray can travel for a given view depth */
vec2 get_max_dir(float view_depth)
{
  float homcco = ProjectionMatrix[2][3] * view_depth + ProjectionMatrix[3][3];
  float max_dist = aoDistance / homcco;
  return vec2(ProjectionMatrix[0][0], ProjectionMatrix[1][1]) * max_dist;
}

vec2 get_ao_dir(float jitter)
{
  /* Only half a turn because we integrate in slices. */
  jitter *= M_PI;
  return vec2(cos(jitter), sin(jitter));
}

void get_max_horizon_grouped(vec4 co1, vec4 co2, vec3 x, float lod, inout float h)
{
  int mip = int(lod) + hizMipOffset;
  co1 *= mipRatio[mip].xyxy;
  co2 *= mipRatio[mip].xyxy;

  float depth1 = textureLod(maxzBuffer, co1.xy, floor(lod)).r;
  float depth2 = textureLod(maxzBuffer, co1.zw, floor(lod)).r;
  float depth3 = textureLod(maxzBuffer, co2.xy, floor(lod)).r;
  float depth4 = textureLod(maxzBuffer, co2.zw, floor(lod)).r;

  vec4 len, s_h;

  vec3 s1 = get_view_space_from_depth(co1.xy, depth1); /* s View coordinate */
  vec3 omega_s1 = s1 - x;
  len.x = length(omega_s1);
  s_h.x = omega_s1.z / len.x;

  vec3 s2 = get_view_space_from_depth(co1.zw, depth2); /* s View coordinate */
  vec3 omega_s2 = s2 - x;
  len.y = length(omega_s2);
  s_h.y = omega_s2.z / len.y;

  vec3 s3 = get_view_space_from_depth(co2.xy, depth3); /* s View coordinate */
  vec3 omega_s3 = s3 - x;
  len.z = length(omega_s3);
  s_h.z = omega_s3.z / len.z;

  vec3 s4 = get_view_space_from_depth(co2.zw, depth4); /* s View coordinate */
  vec3 omega_s4 = s4 - x;
  len.w = length(omega_s4);
  s_h.w = omega_s4.z / len.w;

  /* Blend weight after half the aoDistance to fade artifacts */
  vec4 blend = saturate((1.0 - len / aoDistance) * 2.0);

  h = mix(h, max(h, s_h.x), blend.x);
  h = mix(h, max(h, s_h.y), blend.y);
  h = mix(h, max(h, s_h.z), blend.z);
  h = mix(h, max(h, s_h.w), blend.w);
}

vec2 search_horizon_sweep(vec2 t_phi, vec3 pos, vec2 uvs, float jitter, vec2 max_dir)
{
  max_dir *= max_v2(abs(t_phi));

  /* Convert to pixel space. */
  t_phi /= vec2(textureSize(maxzBuffer, 0));

  /* Avoid division by 0 */
  t_phi += vec2(1e-5);

  jitter *= 0.25;

  /* Compute end points */
  vec2 corner1 = min(vec2(1.0) - uvs, max_dir);  /* Top right */
  vec2 corner2 = max(vec2(0.0) - uvs, -max_dir); /* Bottom left */
  vec2 iter1 = corner1 / t_phi;
  vec2 iter2 = corner2 / t_phi;

  vec2 min_iter = max(-iter1, -iter2);
  vec2 max_iter = max(iter1, iter2);

  vec2 times = vec2(-min_v2(min_iter), min_v2(max_iter));

  vec2 h = vec2(-1.0); /* init at cos(pi) */

  /* This is freaking sexy optimized. */
  for (float i = 0.0, ofs = 4.0, time = -1.0; i < MAX_SEARCH_ITER && time > times.x;
       i++, time -= ofs, ofs = min(exp2(MAX_LOD) * 4.0, ofs + ofs * aoQuality)) {
    vec4 t = max(times.xxxx, vec4(time) - (vec4(0.25, 0.5, 0.75, 1.0) - jitter) * ofs);
    vec4 cos1 = uvs.xyxy + t_phi.xyxy * t.xxyy;
    vec4 cos2 = uvs.xyxy + t_phi.xyxy * t.zzww;
    float lod = min(MAX_LOD, max(i - jitter * 4.0, 0.0) * aoQuality);
    get_max_horizon_grouped(cos1, cos2, pos, lod, h.y);
  }

  for (float i = 0.0, ofs = 4.0, time = 1.0; i < MAX_SEARCH_ITER && time < times.y;
       i++, time += ofs, ofs = min(exp2(MAX_LOD) * 4.0, ofs + ofs * aoQuality)) {
    vec4 t = min(times.yyyy, vec4(time) + (vec4(0.25, 0.5, 0.75, 1.0) - jitter) * ofs);
    vec4 cos1 = uvs.xyxy + t_phi.xyxy * t.xxyy;
    vec4 cos2 = uvs.xyxy + t_phi.xyxy * t.zzww;
    float lod = min(MAX_LOD, max(i - jitter * 4.0, 0.0) * aoQuality);
    get_max_horizon_grouped(cos1, cos2, pos, lod, h.x);
  }

  return h;
}

void integrate_slice(
    vec3 normal, vec2 t_phi, vec2 horizons, inout float visibility, inout vec3 bent_normal)
{
  /* Projecting Normal to Plane P defined by t_phi and omega_o */
  vec3 np = vec3(t_phi.y, -t_phi.x, 0.0); /* Normal vector to Integration plane */
  vec3 t = vec3(-t_phi, 0.0);
  vec3 n_proj = normal - np * dot(np, normal);
  float n_proj_len = max(1e-16, length(n_proj));

  float cos_n = clamp(n_proj.z / n_proj_len, -1.0, 1.0);
  float n = sign(dot(n_proj, t)) * fast_acos(cos_n); /* Angle between view vec and normal */

  /* (Slide 54) */
  vec2 h = fast_acos(horizons);
  h.x = -h.x;

  /* Clamping thetas (slide 58) */
  h.x = n + max(h.x - n, -M_PI_2);
  h.y = n + min(h.y - n, M_PI_2);

  /* Solving inner integral */
  vec2 h_2 = 2.0 * h;
  vec2 vd = -cos(h_2 - n) + cos_n + h_2 * sin(n);
  float vis = saturate((vd.x + vd.y) * 0.25 * n_proj_len);

  visibility += vis;

  /* O. Klehm, T. Ritschel, E. Eisemann, H.-P. Seidel
   * Bent Normals and Cones in Screen-space
   * Sec. 3.1 : Bent normals */
  float b_angle = (h.x + h.y) * 0.5;
  bent_normal += vec3(sin(b_angle) * -t_phi, cos(b_angle)) * vis;
}

float gtao_embree(vec2 uv)
{
  vec4 tx = texelFetch(ao_traceBuffer, ivec2(gl_FragCoord.xy), 0);
  return tx.r;
  //return texture2D(ao_traceBuffer, uv).r;
  //return uv.x * uv.y;
}

void gtao_deferred(
    vec3 normal, vec4 noise, float frag_depth, out float visibility, out vec3 bent_normal)
{
  vec4 tx = texelFetch(ao_traceBuffer, ivec2(gl_FragCoord.xy), 0);
  visibility = tx.x;
}

void gtao(vec3 normal, vec3 position, vec4 noise, out float visibility, out vec3 bent_normal)
{
  //vec4 tx = texture2D(ao_traceBuffer, ivec2(gl_FragCoord.xy));
  vec4 tx = texelFetch(ao_traceBuffer, ivec2(gl_FragCoord.xy), 0);
  visibility = tx.x;
}

/* Multibounce approximation base on surface albedo.
 * Page 78 in the .pdf version. */
float gtao_multibounce(float visibility, vec3 albedo)
{
  if (aoBounceFac == 0.0) {
    return visibility;
  }

  /* Median luminance. Because Colored multibounce looks bad. */
  float lum = dot(albedo, vec3(0.3333));

  float a = 2.0404 * lum - 0.3324;
  float b = -4.7951 * lum + 0.6417;
  float c = 2.7552 * lum + 0.6903;

  float x = visibility;
  return max(x, ((x * a + b) * x + c) * x);
}

/* Use the right occlusion  */
float occlusion_compute(vec3 N, vec3 vpos, float user_occlusion, vec4 rand, out vec3 bent_normal)
{
#ifndef USE_REFRACTION
  if ((int(aoSettings) & USE_AO) != 0) {
    float visibility;
    vec3 vnor = mat3(ViewMatrix) * N;

#  ifdef ENABLE_DEFERED_AO
    gtao_deferred(vnor, rand, gl_FragCoord.z, visibility, bent_normal);
#  else
    gtao(vnor, vpos, rand, visibility, bent_normal);
#  endif

    /* Prevent some problems down the road. */
    visibility = max(1e-3, visibility);

    if ((int(aoSettings) & USE_BENT_NORMAL) != 0) {
      /* The bent normal will show the facet look of the mesh. Try to minimize this. */
      float mix_fac = visibility * visibility * visibility;
      bent_normal = normalize(mix(bent_normal, vnor, mix_fac));

      bent_normal = transform_direction(ViewMatrixInverse, bent_normal);
    }
    else {
      bent_normal = N;
    }

    /* Scale by user factor */
    visibility = pow(visibility, aoFactor);

    return min(visibility, user_occlusion);
  }
#endif

  bent_normal = N;
  return user_occlusion;
}

#define PI 3.14159265359

/**
 * http://amindforeverprogramming.blogspot.de/2013/07/random-floats-in-glsl-330.html?showComment=1507064059398#c5427444543794991219
 */
uint hash3(uint x, uint y, uint z) {
  x += x >> 11;
  x ^= x << 7;
  x += y;
  x ^= x << 3;
  x += z ^ (x >> 14);
  x ^= x << 6;
  x += x >> 15;
  x ^= x << 5;
  x += x >> 12;
  x ^= x << 9;
  return x;
}

/**
 * Generate a random value in [-1..+1).
 * 
 * The distribution MUST be really uniform and exhibit NO pattern at all,
 * because it is heavily used to generate random sample directions for various
 * things, and if the random function contains the slightest pattern, it will
 * be visible in the final image.
 * 
 * In the GLSL world, the function presented in the first answer to:
 * 
 *   http://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
 * 
 * is often used, but that is not a good function, as it has problems with
 * floating point precision and is very sensitive to the seed value, resulting
 * in visible patterns for small and large seeds.
 * 
 * The best implementation (requiring GLSL 330, though) that I found over
 * time is actually this:
 * 
 *   http://amindforeverprogramming.blogspot.de/2013/07/random-floats-in-glsl-330.html
 */
float random(vec2 pos, float time) {
  uint mantissaMask = 0x007FFFFFu;
  uint one = 0x3F800000u;
  uvec3 u = floatBitsToUint(vec3(pos, time));
  uint h = hash3(u.x, u.y, u.z);
  return uintBitsToFloat((h & mantissaMask) | one) - 1.0;
}


/**
 * Generate a uniformly distributed random point on the unit disk oriented around 'n'.
 * 
 * After:
 * http://mathworld.wolfram.com/DiskPointPicking.html
 */
vec3 randomDiskPoint(vec3 rand, vec3 n) {
  float r = rand.x * 0.5 + 0.5; // [-1..1) -> [0..1)
  float angle = (rand.y + 1.0) * PI; // [-1..1] -> [0..2*PI)
  float sr = sqrt(r);
  vec2 p = vec2(sr * cos(angle), sr * sin(angle));
  /*
   * Compute some arbitrary tangent space for orienting
   * our disk towards the normal. We use the camera's up vector
   * to have some fix reference vector over the whole screen.
   */
  vec3 tangent = normalize(rand);
  vec3 bitangent = cross(tangent, n);
  tangent = cross(bitangent, n);
  
  /* Make our disk orient towards the normal. */
  return tangent * p.x + bitangent * p.y;
}


/**
 * Generate a uniformly distributed random point on the unit-sphere.
 * 
 * After:
 * http://mathworld.wolfram.com/SpherePointPicking.html
 */
vec3 randomSpherePoint(vec3 rand) {
  float ang1 = (rand.x + 1.0) * PI; // [-1..1) -> [0..2*PI)
  float u = rand.y; // [-1..1), cos and acos(2v-1) cancel each other out, so we arrive at [-1..1)
  float u2 = u * u;
  float sqrt1MinusU2 = sqrt(1.0 - u2);
  float x = sqrt1MinusU2 * cos(ang1);
  float y = sqrt1MinusU2 * sin(ang1);
  float z = u;
  return vec3(x, y, z);
}

/**
 * Generate a uniformly distributed random point on the unit-hemisphere
 * around the given normal vector.
 * 
 * This function can be used to generate reflected rays for diffuse surfaces.
 * Actually, this function can be used to sample reflected rays for ANY surface
 * with an arbitrary BRDF correctly.
 * This is because we always need to solve the integral over the hemisphere of
 * a surface point by using numerical approximation using a sum of many
 * sample directions.
 * It is only with non-lambertian BRDF's that, in theory, we could sample them more
 * efficiently, if we knew in which direction the BRDF reflects the most energy.
 * This would be importance sampling, but care must be taken as to not over-estimate
 * those surfaces, because then our sum for the integral would be greater than the
 * integral itself. This is the inherent problem with importance sampling.
 * 
 * The points are uniform over the sphere and NOT over the projected disk
 * of the sphere, so this function cannot be used when sampling a spherical
 * light, where we need to sample the projected surface of the light (i.e. disk)!
 */
vec3 randomHemispherePoint(vec3 rand, vec3 n) {
  /**
   * Generate random sphere point and swap vector along the normal, if it
   * points to the wrong of the two hemispheres.
   * This method provides a uniform distribution over the hemisphere, 
   * provided that the sphere distribution is also uniform.
   */
  vec3 v = randomSpherePoint(rand);
  return v * sign(dot(v, n));
}

/**
 * Generate a cosine-weighted random point on the unit hemisphere oriented around 'n'.
 * 
 * @param rand a vector containing pseudo-random values
 * @param n    the normal to orient the hemisphere around
 * @returns    the cosine-weighted point on the oriented hemisphere
 */
vec3 randomCosineWeightedHemispherePoint(vec3 rand, vec3 n) {
  float r = rand.x * 0.5 + 0.5; // [-1..1) -> [0..1)
  float angle = (rand.y + 1.0) * PI; // [-1..1] -> [0..2*PI)
  float sr = sqrt(r);
  vec2 p = vec2(sr * cos(angle), sr * sin(angle));
  /*
   * Unproject disk point up onto hemisphere:
   * 1.0 == sqrt(x*x + y*y + z*z) -> z = sqrt(1.0 - x*x - y*y)
   */
  vec3 ph = vec3(p.xy, sqrt(1.0 - p*p));
  /*
   * Compute some arbitrary tangent space for orienting
   * our hemisphere 'ph' around the normal. We use the camera's up vector
   * to have some fix reference vector over the whole screen.
   */
  vec3 tangent = normalize(rand);
  vec3 bitangent = cross(tangent, n);
  tangent = cross(bitangent, n);
  
  /* Make our hemisphere orient around the normal. */
  return tangent * ph.x + bitangent * ph.y + n * ph.z;
}
