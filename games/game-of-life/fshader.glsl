#version 330 core

uniform vec2 WindowSize;

in vec3 fragColor;
out vec3 color;

void main(){
	//color = vec3(1, gl_FragCoord.x/WindowSize.x, 0);
	color = fragColor;
}
