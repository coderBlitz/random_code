CC := gcc
LIBS := GL GLEW glfw m OpenCL
DEFS := GOL_USE_CL
LIBSFLAG := $(LIBS:%=-l%)
FLAGS := -march=native

%: %.c
	$(CC) -o $@ $^ $(LIBSFLAG) $(FLAGS) $(DEFS:%=-D%)
