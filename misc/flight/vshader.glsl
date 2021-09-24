#version 330 core
layout(location = 0) in vec3 vertexPos_modelspace;
layout(location = 1) in vec3 vertexColor;

out vec3 fragmentColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main(){
	mat4 MVP = projection * view * model;

	gl_Position	= MVP * vec4(vertexPos_modelspace, 1.0);
	fragmentColor = vertexColor;
}
