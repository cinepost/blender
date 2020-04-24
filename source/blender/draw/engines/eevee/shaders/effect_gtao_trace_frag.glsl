/**
 * This shader only compute maximum horizon angles for each directions.
 * The final integration is done at the resolve stage with the shading normal.
 */

uniform float rotationOffset;

#ifdef AO_TRACE_POS

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

  vec4 noise = texelfetch_noise_tex(gl_FragCoord.xy);
  //float rnd = random(gl_FragCoord.xy, noise.y);

  //outWNrm = normalize(transform_direction(ViewMatrixInverse, N));
  outWNrm = randomCosineWeightedHemispherePoint(noise.xyz, normalize(transform_direction(ViewMatrixInverse, N)));
  outWPos = get_world_space_from_depth(uvcoordsvar.xy, texelFetch(depthBuffer, ivec2(gl_FragCoord.xy), 0).r);
}

# else

out vec4 FragColor;

#ifdef DEBUG_AO
uniform sampler2D normalBuffer;

void main()
{
  FragColor = vec4(1.0) * _gtao();
  //FragColor = vec4(1.0) * texelFetch(normalBuffer, ivec2(gl_FragCoord.xy), 0);
}

#else

#    define gtao_depthBuffer depthBuffer
#    define gtao_textureLod(a, b, c) textureLod(a, b, c)

uniform sampler2D normalBuffer;

void main()
{
  FragColor = vec4(1.0) * _gtao();
  //FragColor = vec4(1.0) * texelFetch(normalBuffer, ivec2(gl_FragCoord.xy), 0);
}
#endif
#endif