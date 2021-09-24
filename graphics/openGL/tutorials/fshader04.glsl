#version 330 core

uniform vec2 WindowSize;

in vec3 fragmentColor;
out vec3 color;

void main(){
	color = fragmentColor;
}
