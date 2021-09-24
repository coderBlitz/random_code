#version 330 core
layout(location = 0) in vec2 vertexPosition_modelspace;
layout(location = 1) in vec2 vertex_offset;
layout(location = 2) in vec3 vertex_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main(){
	mat4 MVP = projection*view*model;
	gl_Position = MVP * vec4(vertexPosition_modelspace + vertex_offset, 0.0, 1.0);

	fragColor = vertex_color;
}
