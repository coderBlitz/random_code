#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragmentColor;

void main(){
	mat4 MVP = projection*view*model;
	//mat4 MVP = model*view*projection;
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);

	fragmentColor = vertexColor;
}
