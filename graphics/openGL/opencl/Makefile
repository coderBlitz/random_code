CC := gcc
LIBS := GL GLEW glfw m OpenCL
LIBSFLAG := $(LIBS:%=-l%)

mixed: mixed.c
	$(CC) -o $@ $^ $(LIBSFLAG)
