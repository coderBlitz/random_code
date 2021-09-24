#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<dlfcn.h>

#define PATH_MAX 512

int main(int argc, char *argv[]){
	if(argc > 1){
		printf("Loading '%s'..\n", argv[1]);

		char *path = malloc(PATH_MAX * sizeof(*path));
		getcwd(path, PATH_MAX);
		strcat(path, "/");
		strncat(path, argv[1], PATH_MAX);
		printf("Path (%u): %s\n", strlen(path), path);

		void *lib = dlopen(path, RTLD_LAZY); // Lazy resolves function only when accessed
		if(lib == NULL){
			fprintf(stderr, "Could not open shared library.\n");
			exit(1);
		}

		printf("Attemping to call function testFn..\n");
		void *(*fn)(void) = dlsym(lib, "testFn");
		if(fn == NULL){
			dlclose(lib);
			fprintf(stderr, "Function not found in lib\n");
			free(path);
			exit(1);
		}

		fn();
		printf("Success!\n");

		dlclose(lib);
		free(path);
	}

	return 0;
}
