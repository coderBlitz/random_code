#version 330 core
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertex_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 fragColor;

void main(){
	mat4 MVP = projection * view * model;
	gl_Position = MVP * vec4(vertexPosition_modelspace, 1.0);

	fragColor = vec4(vertex_color, 0.0);
	//fragColor = vec4(gl_InstanceID % 2, gl_VertexID % 2, 0, 1);
	//fragColor = vec4(0.5, gl_VertexID % 2, 0, 1);
	//fragColor = vec4(1,0,1, 0.0);
}
