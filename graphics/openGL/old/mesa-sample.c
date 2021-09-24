#include <stdio.h>
#include <GL/glut.h>
#include <GL/glu.h>

void display(void) {

   glClearColor(0.0, 0.0, 0.0, 0.0);
   glClear(GL_COLOR_BUFFER_BIT);
   glColor3f(1.0,0.5,0.7);
   glutSolidTeapot(3.0);
   glFlush();
}

int main(int argc, char **argv) {
   glutInit(&argc,argv);
   glutInitDisplayMode(GLUT_SINGLE|GLUT_RGBA);
   glutInitWindowPosition(0,0);
   glutInitWindowSize(500,500);
   glutCreateWindow("Test");

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0);
   glMatrixMode(GL_MODELVIEW);

   glutDisplayFunc(display);

   glutMainLoop();
   return 1;
}
