# Important for CL/GL shared buffer
## Include correct headers
```c
#include<GL/glx.h>
#include<CL/cl.h>
#include<CL/cl_gl.h>
```

## Is it supported?
```c
char exts[1024];
ret = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(exts), exts, NULL);
if(!strstr(exts, "cl_khr_gl_sharing")){
	printf("Platform doesn't support CL GL sharing\n");
	glfwTerminate();
	return -3;
}
```

## Necessary properties
```c
	cl_context_properties props[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id,
		CL_GLX_DISPLAY_KHR, (cl_context_properties) glXGetCurrentDisplay(),
		CL_GL_CONTEXT_KHR, (cl_context_properties) glXGetCurrentContext(),
		0
	};
```

Create context with properties, then command queue (may not differ for this case)
```c
	cl_context context = clCreateContext( props, 1, &device_id, NULL, NULL, &ret);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Context creation failed: %d\n", ret);
		glfwTerminate();
		return -2;
	}

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);
```
