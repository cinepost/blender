/**
 * This shader only compute maximum horizon angles for each directions.
 * The final integration is done at the resolve stage with the shading normal.
 */

uniform float rotationOffset;

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
