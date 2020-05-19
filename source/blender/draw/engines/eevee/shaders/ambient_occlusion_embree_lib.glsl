/* aoSettings flags */
#define USE_AO_TRACE 8

uniform sampler2D embreeNormalBuffer;
uniform sampler2D embreeHitsBuffer;

float embree_occlusion() {
  return texelFetch(embreeHitsBuffer, ivec2(gl_FragCoord.xy), 0).r;
  //return texelFetch(embreeNormalBuffer, ivec2(gl_FragCoord.xy), 0).r;
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

/*
 * Geometry functions ***********************************************************
 */

#define TWO_PI        6.2831852
#define FOUR_PI       12.566370
#define INV_PI        0.3183099
#define INV_TWO_PI      0.1591549
#define INV_FOUR_PI     0.0795775
#define EPSILON       0.0001 
#define EQUAL_FLT(a,b,eps)  (((a)>((b)-(eps))) && ((a)<((b)+(eps))))
#define IS_ZERO(a)      EQUAL_FLT(a,0.0,EPSILON)

float seed = 0.; //seed initialized in main
float rnd() { return fract(sin(seed++)*43758.5453123); }

vec2 radialSampleDisk(in vec2 xi) {
    float r = sqrt(1.0 - xi.x);
    float theta = xi.y*TWO_PI;
  return vec2(cos(theta), sin(theta))*r;
}

//https://pdfs.semanticscholar.org/4322/6a3916a85025acbb3a58c17f6dc0756b35ac.pdf
//https://github.com/mmp/pbrt-v3/blob/9f717d847a807793fa966cf0eaa366852efef167/src/core/sampling.cpp#L113
vec2 concentricSampleDisk(in vec2 xi) {
    // Map uniform random numbers to $[-1,1]^2$
    vec2 uOffset = 2. * xi - vec2(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0.0 && uOffset.y == 0.0) return vec2(.0);

    // Apply concentric mapping to point
    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y)) {
        r = uOffset.x;
        theta = (PI/4.0) * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = (PI/2.0) - (PI/4.0) * (uOffset.x / uOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}

vec2 uniformPointWithinCircle( in float radius, in vec2 xi ) {
    vec2 p;
#ifdef CONCENTRIC_DISK
    p = concentricSampleDisk(xi);
#else
    p = radialSampleDisk(xi);
#endif
    p *= radius;
    return p;
}

vec3 uniformDirectionWithinCone( in vec3 d, in float phi, in float sina, in float cosa ) {    
  vec3 w = normalize(d);
    vec3 u = normalize(cross(w.yzx, w));
    vec3 v = cross(w, u);
  return (u*cos(phi) + v*sin(phi)) * sina + w * cosa;
}

//taken from: https://www.shadertoy.com/view/4sSSW3
void basis(in vec3 n, out vec3 f, out vec3 r) {
    if(n.z < -0.999999) {
        f = vec3(0 , -1, 0);
        r = vec3(-1, 0, 0);
    } else {
      float a = 1./(1. + n.z);
      float b = -n.x*n.y*a;
      f = vec3(1. - n.x*n.x*a, b, -n.x);
      r = vec3(b, 1. - n.y*n.y*a , -n.y);
    }
}

mat3 mat3FromNormal(in vec3 n) {
    vec3 x;
    vec3 y;
    basis(n, x, y);
    return mat3(x,y,n);
}

vec3 l2w( in vec3 localDir, in vec3 normal ) {
    vec3 a,b;
    basis( normal, a, b );
  return localDir.x*a + localDir.y*b + localDir.z*normal;
}

void cartesianToSpherical(  in vec3 xyz,
                          out float rho,
                            out float phi,
                            out float theta ) {
    rho = sqrt((xyz.x * xyz.x) + (xyz.y * xyz.y) + (xyz.z * xyz.z));
    phi = asin(xyz.y / rho);
  theta = atan( xyz.z, xyz.x );
}

vec3 sphericalToCartesian( in float rho, in float phi, in float theta ) {
    float sinTheta = sin(theta);
    return vec3( sinTheta*cos(phi), sinTheta*sin(phi), cos(theta) )*rho;
}

vec3 _sampleHemisphereCosWeighted(in vec2 xi) {
#ifdef CONCENTRIC_DISK
    vec2 xy = concentricSampleDisk(xi);
    float r2 = xy.x*xy.x + xy.y*xy.y;
    return vec3(xy, sqrt(max(0.0, 1.0 - r2)));
#else
    float theta = acos(sqrt(1.0-xi.x));
    float phi = TWO_PI * xi.y;
    return sphericalToCartesian( 1.0, phi, theta );
#endif
}

vec3 sampleHemisphereCosWeighted( in vec3 n, in vec2 xi ) {
    return l2w( _sampleHemisphereCosWeighted( xi ), n );
}

vec3 randomDirection( in float Xi1, in float Xi2 ) {
    float theta = acos(1.0 - 2.0*Xi1);
    float phi = TWO_PI * Xi2;
    
    return sphericalToCartesian( 1.0, phi, theta );
}

