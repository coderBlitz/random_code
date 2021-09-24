#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include"shader.c"
#include"bmp.h"

const uint16_t width = 1024;
const uint16_t height = 768;

const GLfloat cube_uv_data[] = {
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 1
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 2
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 3
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 4
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 5
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 6
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 7
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 8
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 9
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 10
	1.0, 0.0,
	0.0, 0.0,
	1.0, 1.0, // End 12
	1.0, 1.0,
	0.0, 1.0,
	0.0, 0.0, // End 12
};

const GLfloat cube_vertex_data[] = {
	-1.0, -1.0, -1.0,
	1.0, -1.0, -1.0,
	-1.0, 1.0, -1.0, // end 1 (back)
	-1.0, 1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, -1.0, -1.0, // end 2
	1.0, -1.0, -1.0,
	1.0, 1.0, -1.0,
	1.0, -1.0, 1.0, // end 3 (right)
	1.0, -1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, -1.0, // end 4
	1.0, 1.0, -1.0,
	-1.0, 1.0, -1.0,
	1.0, 1.0, 1.0, // end 5 (top)
	1.0, 1.0, 1.0,
	-1.0, 1.0, 1.0,
	-1.0, 1.0, -1.0, // end 6
	-1.0, 1.0, -1.0,
	-1.0, -1.0, -1.0,
	-1.0, 1.0, 1.0, // end 7 (left)
	-1.0, 1.0, 1.0,
	-1.0, -1.0, 1.0,
	-1.0, -1.0, -1.0, // end 8
	-1.0, -1.0, -1.0,
	1.0, -1.0, -1.0,
	1.0, -1.0, 1.0, // end 9 (bottom)
	1.0, -1.0, 1.0,
	-1.0, -1.0, -1.0,
	-1.0, -1.0, 1.0, // end 10
	-1.0, -1.0, 1.0,
	1.0, -1.0, 1.0,
	-1.0, 1.0, 1.0, // end 11 (front)
	-1.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, -1.0, 1.0 // end 12
};

GLfloat shift[] = {
	1.0f, 0.0f, 0.0f, 3.0f,
	0.0f, 1.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 1.0f, 3.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat scale[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat rotation[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat projection[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};

void perspective(float *mat, float fovy, float aspect, float zNear, float zFar){
	for(int i = 0;i < 16;++i) mat[i] = 0;

	float f = 1.0/tan(fovy/2.0);
	mat[0] = f/aspect;
	mat[5] = f;

	float diff = zNear - zFar;
	mat[10] = (zFar + zNear)/diff;
	mat[11] = 2*zFar*zNear/diff;
	mat[14] = -1.0;
}

void lookAt(float *mat,
	GLfloat eyeX,
	GLfloat eyeY,
	GLfloat eyeZ,
	GLfloat centerX,
	GLfloat centerY,
	GLfloat centerZ,
	GLfloat upX,
	GLfloat upY,
	GLfloat upZ){
	for(int i = 0;i < 16;++i) mat[i] = 0;

	float F[] = {centerX - eyeX, centerY - eyeY, centerZ - eyeZ};
	float mag = sqrt(F[0]*F[0] + F[1]*F[1] + F[2]*F[2]);
	mat[11] = -mag; // By observation
	float f[] = {F[0]/mag, F[1]/mag, F[2]/mag};

	float up[] = {upX, upY, upZ};
	mag = sqrt(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
	float UP[] = {up[0]/mag, up[1]/mag, up[2]/mag};

	// Cross by un-normalized F by observation
	float s[] = {F[1]*UP[2] - F[2]*UP[1], F[2]*UP[0] - F[0]*UP[2], F[0]*UP[1] - F[1]*UP[0]}; // Cross
	mag = sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
	float S[] = {s[0]/mag, s[1]/mag, s[2]/mag};

	float u[] = {S[1]*f[2] - S[2]*f[1], S[2]*f[0] - S[0]*f[2], S[0]*f[1] - S[1]*f[0]};

	mat[0] = S[0]; // Normalized S by observation
	mat[1] = S[1];
	mat[2] = S[2];

	mat[4] = u[0];
	mat[5] = u[1];
	mat[6] = u[2];

	mat[8] = -f[0];
	mat[9] = -f[1];
	mat[10] = -f[2];

	mat[15] = 1.0;
}

int main(){
	glewExperimental = GL_TRUE;
	if(!glfwInit()){
		fprintf(stderr, "GLFW Failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(width, height, "Totorial 2", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window failed to open\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;
	//glfwSwapInterval(0); // Effectively disables V-sync (60 fps cap)

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed\n");
		return -1;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);


	// GL stuff
	GLuint VertexArrayID; // These 3 are basically a minimum required
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	// Prepare the buffers
	GLuint cubeVertexBuffer;
	glGenBuffers(1, &cubeVertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, cubeVertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertex_data), cube_vertex_data, GL_STATIC_DRAW);

	GLuint uvBuffer;
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_uv_data), cube_uv_data, GL_STATIC_DRAW);

	struct _bmp_file file = readBMP("bee.bmp");
	GLuint textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, file.dib.width, file.dib.height, 0, GL_BGR, GL_UNSIGNED_BYTE, file.data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	GLint *Texture = (int32_t *) file.data;

	// Prepare for loop
	GLuint programID = LoadShaders("vshader05.glsl", "fshader05.glsl");
	printf("ProgramID: %u\n", programID);
	if(programID == 0){
		fprintf(stderr, "Program failed to load correctly.\n");
		return -3;
	}

	//lookAt(rotation, 4,3,3, 0,0,0, 0,1,0);
	perspective(projection, M_PI/4.0, (float)4.0/3.0, 0.1f, 100.0f);

	//for(int i = 0;i < 16;++i) printf("%f%c", rotation[(i/4)+(i%4)*4], ((i&3) == 3)?'\n':' ');
	//printf("\n");

	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");


	register int count = 0;
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	do{
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		lookAt(rotation, 6*cos(count*M_PI/180.0),2,6*sin(count*M_PI/180.0), 0,0,0, 0,1,0);

		glUseProgram(programID);

		// Pass vars to shaders
		glUniform2f(uniform_WindowSize, width, height);
		glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale);
		glUniformMatrix4fv(uniform_view, 1, GL_TRUE, rotation);
		glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

		/***** DRAW STUFF *****/
		glEnableVertexAttribArray(0); // Start cube
		glBindBuffer(GL_ARRAY_BUFFER, cubeVertexBuffer);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glEnableVertexAttribArray(1); // UV
		glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);

		glDrawArrays(GL_TRIANGLES, 0, 12*3);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0); // End cube
		/***** END DRAW STUFF *****/

		GLenum ret = 0;
		if(ret = glGetError() != GL_NO_ERROR){
			fprintf(stderr, "Error in main loop %d\n", ret);
			break;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
		++count;
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	glfwTerminate();
	free(file.data);

	return 0;
}
