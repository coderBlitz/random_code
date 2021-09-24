#version 330 core

uniform vec2 WindowSize;
uniform sampler2D myTextureSampler;

in vec2 UV;
in vec3 fragmentColor;
out vec3 color;

void main(){
	color = texture(myTextureSampler, UV).rgb;
}
