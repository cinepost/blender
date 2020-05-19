/**
 * This shader only compute maximum horizon angles for each directions.
 * The final integration is done at the resolve stage with the shading normal.
 */

uniform float rotationOffset;

#ifdef AO_TRACE_POS

// prepas for embree tracer
layout(location = 0) out mediump vec3 outWNrm;
layout(location = 1) out highp vec3 outWPos;

in vec4 rand;
in vec4 uvcoordsvar;

in vec3 viewPosition;
in vec3 worldPosition;
in vec3 worldNormal;
in vec3 worldNormalFlat;

uniform float sampleNum;
uniform sampler2D normalBuffer; // ssr normal on samples > 0

void main() {
  //vec3 true_normal = normalize(cross(dFdx(viewPosition), dFdy(viewPosition)));
  vec4 rand = texelfetch_noise_tex(gl_FragCoord.xy);
  
  //vec4 rand = texelFetch(utilTex, ivec3(ivec2(gl_FragCoord.xy) % LUT_SIZE, 2.0), 0);

  if ((sampleNum > 1) && (aoUseBump > 0)) {
    // Using ssr normal
    vec3 N = normal_decode(texelFetch(normalBuffer, ivec2(gl_FragCoord.xy), 0).rg, viewCameraVec);
    vec3 nN = normalize(transform_direction(ViewMatrixInverse, N));
    outWNrm = randomHemispherePoint(rand.xyz, nN);
  } else {
    // Using our normal
    outWNrm = randomHemispherePoint(rand.xyz, worldNormalFlat);
  }
  outWPos = worldPosition;
}

# else

out vec4 FragColor;

#ifdef DEBUG_AO

#    define gtao_depthBuffer depthBuffer
#    define gtao_textureLod(a, b, c) textureLod(a, b, c)

in vec4 uvcoordsvar;

void main() {
  vec2 uvs = saturate(gl_FragCoord.xy / vec2(textureSize(gtao_depthBuffer, 0).xy));
  float depth = gtao_textureLod(gtao_depthBuffer, uvs, 0.0).r;

  if (depth == 1.0) {
    /* to look like the builtin one */
    FragColor = vec4(0.0);
    return;
  }

  FragColor = vec4(1.0) * embree_occlusion();
}

#else

in vec4 uvcoordsvar;

void main() {

  FragColor = vec4(1.0) * embree_occlusion();
}
#endif
#endif