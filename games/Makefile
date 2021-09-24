LIBS=-lGL -lGLU -lGLEW -lglfw -lm -lpthread
FLAGS := -O3

%: %.c
	gcc -o $@ ${LIBS} $(FLAGS) $<
