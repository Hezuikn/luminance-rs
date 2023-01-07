in vec3 v_color;

layout out vec3 frag_color;
layout out float frag_white;

void main() {
  frag_color = v_color.rgb;
  frag_white = 1.;
}
