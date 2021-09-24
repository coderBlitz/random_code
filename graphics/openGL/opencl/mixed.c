#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<string.h>
#include<GL/glew.h>
#include<GL/glx.h>
#include<GLFW/glfw3.h>

#define CL_TARGET_OPENCL_VERSION 220
#include<CL/cl.h>
#include<CL/cl_gl.h>
#include"shader.c"

#define MAX_SOURCE_SIZE (0x10000)

const uint16_t width = 1024;
const uint16_t height = 768;

const GLfloat vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	1.0f, -1.0f, 0.0f,
	0.0f, 1.0f, 0.0f
/*	1.0f, 1.0f, 0.0f,
	-1.0f, 1.0f, 0.0f,
	0.0f, -1.0f, 0.0f*/
};

const GLfloat scale[] = {
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

GLFWwindow *initWindow(void){
	glewExperimental = GL_TRUE;
	if(!glfwInit()){
		fprintf(stderr, "GLFW Failed\n");
		return NULL;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(width, height, "Totorial 2", NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "Window failed to open\n");
		glfwTerminate();
		return NULL;
	}

	glfwMakeContextCurrent(window);
	glewExperimental = GL_TRUE;

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed\n");
		glfwTerminate();
		return NULL;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

	return window;
}



int main(){
	GLFWwindow *window = initWindow();
	if(window == NULL){
		fprintf(stderr, "Window failed to open\n");
		glfwTerminate();
		return -1;
	}
	//glfwSwapInterval(0);

	// Must load shaders before opencl context, otherwise shader errors (probably from nvidia card)
	GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl");
	printf("ProgramID: %u\n", programID);
	glUseProgram(programID);

	// Load the opencl kernel source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen("vector_add_kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		glfwTerminate();
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose( fp );

	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;   
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

	// Check for support
	char exts[1024];
	ret = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(exts), exts, NULL);
	if(!strstr(exts, "cl_khr_gl_sharing")){
		printf("Platform doesn't support CL GL sharing\n");
		glfwTerminate();
		return -3;
	}

	// Get properties, create context and queue
	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id,
		CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
		CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
		0
	};
	cl_context context = clCreateContext( props, 1, &device_id, NULL, NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Context creation failed: %d\n", ret);
		glfwTerminate();
		return -2;
	}
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Failed to build program\n");
		return -4;
	}
	free(source_str);
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Kernel creation failed\n");
		return -4;
	}


	// GL stuff
	GLuint VertexArrayID; // These 3 are basically a minimum required
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);


	// Create and load vertices into buffer
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	// Create memory buffers on the device for each vector
	cl_mem vertex_mem_obj = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE, vertexbuffer, &ret);

	// Set the argument(s) of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&vertex_mem_obj);

	lookAt(rotation, 2,1,3, 0,0,0, 0,1,0);
	perspective(projection, M_PI/4.0, (float)4.0/3.0, 0.1f, 100.0f);

	GLint uniform_WindowSize = glGetUniformLocation(programID, "WindowSize"); // Used to pass windows size
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");

	// Pass vars to shaders
	glUniform2f(uniform_WindowSize, width, height);
	glUniformMatrix4fv(uniform_model, 1, GL_FALSE, scale);
	glUniformMatrix4fv(uniform_view, 1, GL_TRUE, rotation);
	glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

	size_t global_size = 9;
	size_t item_size = 1; // Use 64 as laptop max, 1024 as gpu max
	do{
		glClearColor(0.0f, 0.0f, 0.4f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ret = clEnqueueAcquireGLObjects(command_queue, 1, &vertex_mem_obj, 0, NULL, NULL);
		if(ret){
			printf("Could not acquire GL objects: %d\n", ret);
			if(ret == CL_INVALID_CONTEXT) printf("Context\n");
		}else{
			ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &item_size, 0, NULL, NULL);
			if(ret){
				printf("Kernel failed\n");
				break;
			}
			ret = clEnqueueReleaseGLObjects(command_queue, 1, &vertex_mem_obj, 0, NULL, NULL);
		}

		/***** DRAW STUFF *****/
		glDrawArrays(GL_TRIANGLES, 0, 3);
		/***** END DRAW STUFF *****/

		if(ret = glGetError() != GL_NO_ERROR){
			fprintf(stderr, "GL Error: %d\n", ret);
			break;
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	clFlush(command_queue);
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(vertex_mem_obj);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	glfwTerminate();

	return 0;
}
