#version 330 core
layout(location = 0) in vec2 vertexPosition_modelspace;
layout(location = 1) in vec2 vertex_offset;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;

void main(){
	mat4 MVP = projection*view*model;
	//mat4 MVP = model*view*projection;
	/*	Note:
		Add offset after to avoid the MVP affecting the offset. Only valid for static 2-D view.
		If rotation/scaling desired, then add directly to modelspace. Then change code in main file.
	*/
	//gl_Position = MVP * vec4(vertexPosition_modelspace, 0.0, 1.0) + vec4(vertex_offset, 0, 0);
	gl_Position = MVP * vec4(vertexPosition_modelspace + vertex_offset, 0.0, 1.0);

	if((gl_InstanceID % 2) == 0){
		fragColor = vec3(0.7, gl_VertexID % 2, 0);
	}else{
		fragColor = vec3(0, gl_VertexID % 2, 0.7);
	}
	/*if(vertex_offset.x == 0){
		fragColor = vec3(1.0, 0.0, 0.0);
	}else if(vertex_offset.x < 0){
		fragColor = vec3(0.0, 1.0, 0.0);
	}else if(vertex_offset.x > 0){
		fragColor = vec3(0.0, 0.0, 1.0);
	}*/
}
