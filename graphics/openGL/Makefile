CC := gcc
LIBS := GL GLEW glfw m stl
LIBSFLAG := $(LIBS:%=-l%)
FLAGS :=

%: %.c
	$(CC) -o $@ $^ $(LIBSFLAG) $(FLAGS)
