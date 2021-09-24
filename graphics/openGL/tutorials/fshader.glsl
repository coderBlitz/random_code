#version 330 core

uniform vec2 WindowSize;
out vec3 color;

void main(){
	color = vec3(1, gl_FragCoord.x/WindowSize.x, 0);

}
