CC := gcc
LIBS := m GL GLEW glfw

loiter: loiter.c shader.c helper.h
	gcc -o $@ $< ${LIBS:%=-l%}
