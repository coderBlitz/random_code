/* Description: Logic used to give an programmed entity movement closer to that
				 of an actual aircraft, specifically when loitering.
				Though parts could be adapted for other manners of flight.

	TODO:
	- Figure out when to rotate icon in matrix operations
*/

#include<math.h>
#include<stdint.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<time.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>

#include"helper.h"
#include"shader.c"

#define TORAD (M_PI/180.0)

const uint16_t width = 1024;
const uint16_t height = 768;

const float meters_per_pixel = 3.0;

struct position_t{
	double X;
	double Y;
	double Z;
};
struct entity_t{
	struct position_t pos;
	double velocity;
	double bearing;		// Heading (degrees)
	double mag;			// Max turn rate (degrees per second)
};

GLfloat I[] = {
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, 0.0, 0.0, 1.0
};
GLfloat model[] = {
	0.02f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.02f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat view[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};
GLfloat projection[] = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
};

const float ent_icon_vertex[] = {
	-0.5, -1.0, 0.0,
	0.5, -1.0, 0.0,
	0.0, 1.0, 0.0
};
const float ent_icon_color[] = {
	1.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, 0.0
};

#define CENTER_LEN 30
float cent_icon_vertex[CENTER_LEN];
float cent_icon_color[CENTER_LEN];

float radius_path[3*36]; // XYZ for N points (N=36 currently)
float radius_color[3*36];

/********** Helper functions **********/
void updatePosMat(float *mat, float x, float y, float z){
	mat[3] = x;
	mat[7] = y;
	mat[11] = z;
}
void updateRotMat(float *mat, float radians){
	mat[0] = -cos(radians); // Works through experimentation
	mat[1] = sin(radians);
	mat[5] = -mat[0];
	mat[4] = mat[1];
}
void updateScaleMat(float *mat, float Xscale, float Yscale, float Zscale){
	mat[0] = Xscale;
	mat[5] = Yscale;
	mat[10] = Zscale;
}

GLFWwindow *windowInit(char *title){
	//glewExperimental = GL_TRUE;
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed\n");
		return NULL;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow *window = glfwCreateWindow(width, height, title, NULL, NULL);
	if(window == NULL){
		fprintf(stderr, "GLFW could not create window\n");
		glfwTerminate();
		return NULL;
	}
}

GLuint createBuffer(const void *data, size_t data_size){
	GLuint buffer;
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, data_size, data, GL_STATIC_DRAW);

	return buffer;
}

// Calculate and update entity to loiter target at radius. Return distance
double calculateLoiter(struct entity_t *entity, struct entity_t *target, double radiusM, double prev_distance, double dt){
	struct position_t dir = {entity->pos.X - target->pos.X, entity->pos.Y - target->pos.Y, entity->pos.Z - target->pos.Z};
	struct position_t ent_vector = {sin(entity->bearing*TORAD), cos(entity->bearing*TORAD), 0.0};
	struct position_t ent_math_vector = {cos(entity->bearing*TORAD), -sin(entity->bearing*TORAD), 0.0};

	double db = 0.0; // Change in bearing (independent var)

	double radius_min = (180.0/entity->mag) * entity->velocity/M_PI; // (time to turn 360 degrees)*speed = 2*pi*radius
	if(radiusM < radius_min) radiusM = radius_min;

	entity->pos.X += ent_vector.X * entity->velocity * dt;
	entity->pos.Y += ent_vector.Y * entity->velocity * dt;

	double cross = dir.X*ent_math_vector.Y - dir.Y*ent_math_vector.X;
	double lr_cross = dir.X*ent_vector.Y - dir.Y*ent_vector.X;

	double distance = sqrt(dir.X*dir.X + dir.Y*dir.Y); // Distance from center
	double dd = distance - prev_distance;
	double diff = distance - radiusM;
	double mag = entity->mag;

	/******** This logic should work (mostly in this order) *******
		If greatly outside radius, turn towards center
		If outside radius and increasing distance, turn inward
		If outside radius, elminate turns toward outside
		If inside radius, but near, reduce turn-in rate
		If inside radius, but near and decreasing distance, turn outward
		Else defaults to turning in nearest direction, or not (depends how large radius is)
	*******/

	// Determines clockwise or counter-clockwise
	if(cross > 0) db = mag;
	else db = -mag;

	// If smallest radius possible is inside target radius, don't turn at all
	if(radiusM > 2*radius_min){
		if(distance < (radiusM - radius_min)) db = 0.0;
	}

	if(diff > radius_min){
		if(lr_cross > 0){
			db = -mag;
		}else if(lr_cross < 0){
			db = mag;
		}
	}
	else if(diff > 0){
		if(dd > -entity->velocity/entity->mag/2){
			if(lr_cross > 0)
				db = -mag;
			else
				db = mag;
		}else{ // If decreasing distance (below threshold of if-statement)
			if(lr_cross > 0 && cross > 0){ // If turning outward
				//db *= 0.8; // How much to adjust turn out rate when heading toward radius
			}else if(lr_cross < 0 && cross < 0){
				//db *= 0.8;
			}
		}
	}else if(diff > -entity->velocity){
		if(lr_cross > 0){
			if(dd < 0)
				db = mag;
			else if(db < 0)
				db *= 0.2; // Rate at which to turn inward, when close to radius
		}else if(lr_cross < 0){
			if(dd < 0)
				db = -mag;
			else if(db > 0)
				db *= 0.2;
		}
	}

	entity->bearing += db*dt;

	return distance; // Return distance to avoid re-computing in main
}

void drawBuffers(GLuint vertices, GLuint colors, size_t num_vertices, GLenum method){
	// First the vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertices);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

	// Then the colors
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colors);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	// Then draw and disable
	glDrawArrays(method, 0, num_vertices);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
}

// Create XYZ points for inscribed N-sided polygon. Sets uniform Z value
void tracePolygon(uint32_t N, float *array, float Z){
	for(int i = 0;i <= N;i++){
		array[3*i] = cos(2*i*M_PI/N);
		array[3*i+1] = sin(2*i*M_PI/N);
		array[3*i+2] = Z;
	}
}

/**************************/
/********** MAIN **********/
/**************************/
struct entity_t ent = {{1000.0, 1000.0, 0.0}, 150.0, 0.0, 10.0}; // Entity and target to use
struct entity_t center = {{0.0, 0.0, 0.0}, 15.0, 90.0, 5.0};

int main(){
	srand(time(0));
/* Note:	The turn directions here are compass-based, so positive turn rate
			 is indicative of a clockwise turn, and vice-versa.
*/

	/// Initialize GL and window
	GLFWwindow *window = windowInit("Loiter");
	if(window == NULL){
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

	if(glewInit() != GLEW_OK){
		fprintf(stderr, "Glew failed to init");
		return -1;
	}

	GLuint programID = LoadShaders("vshader.glsl", "fshader.glsl");
	if(programID == 0){
		fprintf(stderr, "Program failed to load shaders\n");
		return -2;
	}
	GLuint vertexArrayID;
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);


	/// Initialize vertex arrays
	// Create center "dot" with triangle fan
	cent_icon_vertex[0] = 0.0;
	cent_icon_vertex[1] = 0.0;
	cent_icon_vertex[2] = 0.0;
	tracePolygon(CENTER_LEN/3-2, &cent_icon_vertex[3], 0.0); // N-2 because the first 2 vertices are center and start/end
	
	for(int i = 0;i < CENTER_LEN;i++) cent_icon_color[i] = 0.3;

	tracePolygon(36, radius_path, 0.0);
	for(int i = 0;i < 36;i++){
		radius_color[3*i] = 1.0;
		radius_color[3*i+1] = 0.0;
		radius_color[3*i+2] = 0.0;
	}

	/// Buffers
	GLuint entVertexBuffer = createBuffer(ent_icon_vertex, sizeof(ent_icon_vertex));
	GLuint entColorBuffer = createBuffer(ent_icon_color, sizeof(ent_icon_color));

	GLuint centVertexBuffer = createBuffer(cent_icon_vertex, sizeof(cent_icon_vertex));
	GLuint centColorBuffer = createBuffer(cent_icon_color, sizeof(cent_icon_color));

	GLuint radiusBuffer = createBuffer(radius_path, sizeof(radius_path));
	GLuint radiusColorBuffer = createBuffer(radius_color, sizeof(radius_color));

	/// Loop variables
	unsigned long delay_usec = 16000;
	double delay_fraction = delay_usec / 1e6;

	double distance = 0.0; // Distance from center
	double prev_distance = 0.0;
	double point_db = 0.0;

	double radius_min = (180.0/ent.mag)*ent.velocity/M_PI; // (time to turn 360 degrees)*speed = 2*pi*radius
	double radiusM = 15.0;
	if(radiusM < radius_min) radiusM = radius_min;
	printf("Radius min: %.4lf\tRadius target: %.4lf\n", radius_min, radiusM);

	unsigned long count = 0;
	unsigned long N = 300/delay_fraction;

	/// More GL variables
	GLuint uniform_model = glGetUniformLocation(programID, "model");
	GLuint uniform_view = glGetUniformLocation(programID, "view");
	GLuint uniform_projection = glGetUniformLocation(programID, "projection");

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glUseProgram(programID);

	/// Event loop
	do{
		++count;
		glClearColor(0.0, 0.3, 0.0, 0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Update entity and associated matrices
		prev_distance = distance;
		distance = calculateLoiter(&ent, &center, radiusM, prev_distance, delay_fraction);

		updatePosMat(model, ent.pos.X/width/meters_per_pixel, ent.pos.Y/height/meters_per_pixel, 0);
		updateRotMat(view, ent.bearing*TORAD);
		updateScaleMat(model, 0.02, 0.02, 1.0);

		//printf("%8.4lf,%8.4lf,%8.4lf,%8.4lf,%8.4lf\n", ent.X, ent.Y, distance, center.X, center.Y); // Terminal output
		// Manually update center, since random movement (i.e has no target/destination)
		point_db = 2.0 * center.mag * (float)rand()/RAND_MAX - center.mag; // Should give random direction within range
		center.bearing += point_db;
		center.pos.X += center.velocity * cos(center.bearing*TORAD) * delay_fraction;
		center.pos.Y += center.velocity * sin(center.bearing*TORAD) * delay_fraction;

		/*** Draw ***/
		glUniformMatrix4fv(uniform_model, 1, GL_TRUE, view); // Rotation relative to vertex (icon) if multiplied first
		glUniformMatrix4fv(uniform_view, 1, GL_TRUE, model);
		glUniformMatrix4fv(uniform_projection, 1, GL_TRUE, projection);

		// Draw entity icon
		drawBuffers(entVertexBuffer, entColorBuffer, 3, GL_TRIANGLES);

		// Updating transform matrix for center/target
		updatePosMat(model, center.pos.X/width/meters_per_pixel, center.pos.Y/height/meters_per_pixel, 0);
		glUniformMatrix4fv(uniform_model, 1, GL_TRUE, model);
		glUniformMatrix4fv(uniform_view, 1, GL_FALSE, I);

		// Draw center
		drawBuffers(centVertexBuffer, centColorBuffer, (CENTER_LEN/3), GL_TRIANGLE_FAN);

		// Update matrix for radius, then draw
		updateScaleMat(model, radiusM/width/meters_per_pixel, radiusM/height/meters_per_pixel, 1.0);
		glUniformMatrix4fv(uniform_model, 1, GL_TRUE, model);
		drawBuffers(radiusBuffer, radiusColorBuffer, 36, GL_LINE_LOOP);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);


	// Cleanup
	glfwTerminate();

	return 0;
}
