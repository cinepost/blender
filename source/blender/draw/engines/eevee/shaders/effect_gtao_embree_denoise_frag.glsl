layout(std140) uniform denoise_block
{
  float stepwidth;
  float c_phi;
  float n_phi;
  float p_phi;
};

uniform sampler2D aoEmbreeRawBuffer;
uniform sampler2D wnormBuffer;
uniform sampler2D wposBuffer;

out vec4 FragColor;

#define KERNEL_SIZE 25

//uniform float kernel[KERNEL_SIZE];
//uniform ivec2 offset[KERNEL_SIZE];

//in vec4 uvcoordsvar;

void main(void) {
  const float kernel[25] = float[25](
    1.0/256.0, 1.0/64.0, 3.0/128.0, 1.0/64.0, 1.0/256.0,
    1.0/64.0,  1.0/16.0, 3.0/32.0,  1.0/16.0, 1.0/64.0,
    3.0/128.0, 3.0/32.0, 9.0/64.0,  3.0/32.0, 3.0/128.0,
    1.0/64.0,  1.0/16.0, 3.0/32.0,  1.0/16.0, 1.0/64.0,
    1.0/256.0, 1.0/64.0, 3.0/128.0, 1.0/64.0, 1.0/256.0 );

  const ivec2 offset[25] = ivec2[25]( 
    ivec2(-2,-2), ivec2(-1,-2), ivec2(0,-2), ivec2(1,-2), ivec2(2,-2), 
    ivec2(-2,-1), ivec2(-1,-1), ivec2(0,-2), ivec2(1,-1), ivec2(2,-1), 
    ivec2(-2, 0), ivec2(-1, 0), ivec2(0, 0), ivec2(1, 0), ivec2(2, 0), 
    ivec2(-2, 1), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1), ivec2(2, 1),
    ivec2(-2, 2), ivec2(-1, 2), ivec2(0, 2), ivec2(1, 2), ivec2(2, 2) );

  vec3 sum = vec3(0.0);
  ivec2 tx = ivec2(gl_FragCoord.xy);
  
  vec3 cval = vec3(1.0) * texelFetch(aoEmbreeRawBuffer, tx, 0).r;
  vec3 nval = texelFetch(wnormBuffer, tx, 0).xyz;
  vec3 pval = texelFetch(wposBuffer, tx, 0).xyz;
  
  float cum_w = 0.0;
  
  for(int i = 0; i < KERNEL_SIZE; i++) {
    ivec2 crd = tx + ivec2(offset[i] * stepwidth);
    //if (crd == tx)
    //  continue;
    
    // Color (ao)
    vec3 ctmp = vec3(1.0) * texelFetch(aoEmbreeRawBuffer, crd, 0).r;
    vec3 t = cval - ctmp;
    float dist2 = dot(t, t);
    float c_w = min(exp(-(dist2)/c_phi), 1.0); c_w = 1.0;

    // Normal
    vec3 ntmp = texelFetch(wnormBuffer, crd, 0).xyz;
    t = nval - ntmp;
    dist2 = max(dot(t,t)/(stepwidth*stepwidth),0.0);
    //dist2 = dot(t,t);
    float n_w = min(exp(-(dist2)/n_phi), 1.0);
    //n_w = max(0.0, dot(nval, ntmp)/(stepwidth*stepwidth));
    //n_w *= n_w;
    //n_w = 1.0;

    // Pos
    vec3 ptmp = texelFetch(wposBuffer, crd, 0).xyz;
    t = pval - ptmp;
    //dist2 = dot(t,t);
    dist2 = max(dot(t,t)/(stepwidth*stepwidth),0.0);
    float p_w = min(exp(-(dist2)/p_phi),1.0);

    float weight = c_w * min(n_w, p_w);
    sum += ctmp * weight * kernel[i];
    cum_w += weight * kernel[i];
  }

  FragColor = vec4(sum / cum_w, 1.0);
}
