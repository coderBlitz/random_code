#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<CL/cl.h>

int main(int argc, char *argv[]){
	int N = 5;
	cl_platform_id platforms[N];
	cl_device_id devices[N];
	cl_uint num_plats;
	cl_uint num_devs;
	cl_uint ret;

	ret = clGetPlatformIDs(N, platforms, &num_plats);
	if(ret != CL_SUCCESS){
		fprintf(stderr, "Getting platforms failed.\n");
		return -1;
	}

	char version[40];
	char vendor[50];
	size_t vendor_len = 0;
	char name[50];
	size_t name_len = 0;
	for(int i = 0;i < num_plats;++i){
		ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, N, devices, &num_devs);
		if(ret != CL_SUCCESS){
			fprintf(stderr, "Getting devices failed.\n");
			continue;
		}
		ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, &vendor_len);
		if(ret != CL_SUCCESS){
			fprintf(stderr, "Failed to query platform %d vendor\n", i);
		}
		ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, &name_len);
		if(ret != CL_SUCCESS){
			fprintf(stderr, "Failed to query platform %d name\n", i);
		}
		ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
		if(ret != CL_SUCCESS){
			fprintf(stderr, "Failed to query platform %d name\n", i);
		}

		printf("Platform %d: %s, %s (%s)\n", i, vendor, name, version);


		cl_device_type type;
		char type_str[10];
		for(int j = 0;j < num_devs;++j){
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Failed to query device %d info\n", j);
			}
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor), vendor, &vendor_len);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Failed to query device %d info\n");
			}
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, &name_len);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Failed to query device %d info\n", j);
			}

			switch(type){
				case CL_DEVICE_TYPE_CPU:
					strcpy(type_str, "CPU");
					break;
				case CL_DEVICE_TYPE_GPU:
					strcpy(type_str, "GPU");
					break;
				default:
					strcpy(type_str, "Unknown");
					break;
			}

			char exts[512];
			ret = clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, sizeof(exts), exts, &name_len);
			if(ret != CL_SUCCESS){
				fprintf(stderr, "Failed to query device %d info\n", j);
			}

			printf("  Device %i: %s, %s (%s) (ext: %s)\n", j, vendor, name, type_str, exts);
		}
	}

	return 0;
}
