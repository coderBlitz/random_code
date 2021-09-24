#include <GLFW/glfw3.h>
void render_loop()
{
    glClearColor ( .7, .1, .1, 1.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glLoadIdentity();
    glViewport(0,0,1024,768);
    glMatrixMode(GL_PROJECTION);
    //gluPerspective( 65.0, (double)1024/(double)768, 1.0, 60.0 );
    glOrtho(0,1024,768,0,100,-100);
//    glMatrixMode(GL_MODELVIEW);

    glPointSize(10);
    glBegin(GL_POINTS);
    glColor4f(1,1,1,1);
    glVertex3f(512,384,0);
    glEnd();
}
int main ( int argc, char* argv[] )
{
    //init glfw
    glfwInit();

	glfwWindowHint(GLFW_SAMPLES, 4);
    GLFWwindow *window = glfwCreateWindow ( 1024, 768, "Test", NULL, NULL );
	glfwMakeContextCurrent(window);

    do {
        render_loop();
	    glfwSwapBuffers(window);
    } while ( glfwWindowShouldClose(window) == 0);

    glfwTerminate();
}
