/**
 * This shader only compute maximum horizon angles for each directions.
 * The final integration is done at the resolve stage with the shading normal.
 */

uniform float rotationOffset;
uniform float sampleNum;

#ifdef AO_TRACE_POS

// prepas for embree tracer
layout(location = 0) out mediump vec3 outWNrm;
layout(location = 1) out highp vec3 outWPos;

in vec4 rand;
in vec4 uvcoordsvar;
in vec3 viewPosition;

in vec3 worldNormal;
in vec3 worldPosition;

in vec3 viewNormal;

void main() {
  vec4 rand = texelfetch_noise_tex(gl_FragCoord.xy);
  //vec4 rand = texelFetch(utilTex, ivec3(ivec2(gl_FragCoord.xy) % LUT_SIZE, 2.0), 0);

  outWNrm = randomHemispherePoint(rand.xyz, worldNormal);
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