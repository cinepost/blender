/**
 * This shader only compute maximum horizon angles for each directions.
 * The final integration is done at the resolve stage with the shading normal.
 */

uniform float rotationOffset;

#ifdef AO_TRACE_POS

// prepas for embree tracer
layout(location = 0) out vec3 outWNrm;
layout(location = 1) out vec3 outWPos;

in vec4 rand;
in vec4 uvcoordsvar;
in vec3 viewPosition;
in vec3 worldNormal;
uniform sampler2D normalBuffer;

void main()
{
  vec3 V = viewCameraVec;
  vec3 N = normal_decode(texelFetch(normalBuffer, ivec2(gl_FragCoord.xy), 0).rg, V);
  vec3 nN = normalize(transform_direction(ViewMatrixInverse, N));
  vec4 noise = texelfetch_noise_tex(gl_FragCoord.xy);
  
  outWNrm = randomCosineWeightedHemispherePoint(noise.xyz, nN);
  outWPos = get_world_space_from_depth(uvcoordsvar.xy, texelFetch(depthBuffer, ivec2(gl_FragCoord.xy), 0).r);
}

# else

out vec4 FragColor;

#ifdef DEBUG_AO

in vec4 uvcoordsvar;
uniform sampler2D normalBuffer;

void main()
{
  FragColor = vec4(1.0) * gtao_embree(uvcoordsvar.xy);
}

#else

#    define gtao_depthBuffer depthBuffer
#    define gtao_textureLod(a, b, c) textureLod(a, b, c)

in vec4 uvcoordsvar;
uniform sampler2D normalBuffer;

void main()
{
  FragColor = vec4(1.0) * gtao_embree(uvcoordsvar.xy);
}
#endif
#endif