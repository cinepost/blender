/* aoSettings flags */
#define USE_AO_TRACE 8

uniform sampler2D aoEmbreeBuffer;

float embree_occlusion() {
  return texelFetch(aoEmbreeBuffer, ivec2(gl_FragCoord.xy), 0).r;
}

void gtao_deferred_embree(vec3 normal, vec4 noise, float frag_depth, out float visibility, out vec3 bent_normal)
{
  visibility = embree_occlusion();
  bent_normal = normal;
}

void gtao_embree(vec3 normal, vec3 position, vec4 noise, out float visibility, out vec3 bent_normal)
{
  visibility = embree_occlusion();
  bent_normal = normal;
}

/* Use the right occlusion  */
float occlusion_compute_embree(vec3 N, vec3 vpos, float user_occlusion, vec4 rand, out vec3 bent_normal) {
#ifndef USE_REFRACTION
  if ((int(aoSettings) & USE_AO) != 0) {
    float visibility;
    vec3 vnor = mat3(ViewMatrix) * N;

#  ifdef ENABLE_DEFERED_AO
    gtao_deferred_embree(vnor, rand, gl_FragCoord.z, visibility, bent_normal);
#  else
    gtao_embree(vnor, vpos, rand, visibility, bent_normal);
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
  float ang1 = rand.x * 2.0 * PI; // [-1..1) -> [0..2*PI)
  float u = 2.0 * rand.y - 1.0; // [-1..1), cos and acos(2v-1) cancel each other out, so we arrive at [-1..1)
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
