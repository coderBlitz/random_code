#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<unistd.h>
#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<glm/gtc/matrix_transform.hpp>
#include"controls.hpp"

glm::mat4 Projection;
glm::mat4 View;

glm::vec3 position = glm::vec3(0, 0, 5);
float horizontalAngle = 3.14f;
float verticalAngle = 0.0f;
float initialFoV = 45.0f;

float speed = 3.0f;
float mouseSpeed = 0.05f;
double currentTime = -1.0, lastTime = -1.0;

void computeMatricesFromInputs(GLFWwindow *window){
	// Get time things
	if((int)lastTime == -1){
		lastTime = glfwGetTime();
		return;
	}else currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);

	// Get the cursor movement
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Reset cursor movement
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	glfwSetCursorPos(window, width/2, height/2);

	horizontalAngle += mouseSpeed * deltaTime * float(width/2 - xpos);
	verticalAngle += mouseSpeed * deltaTime * float(height/2 - ypos);

	glm::vec3 direction(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
	);

	glm::vec3 right = glm::vec3(
		sin(horizontalAngle - 3.14f/2.0f),
		0,
		cos(horizontalAngle - 3.14f/2.0f)
	);

	glm::vec3 up = glm::cross( right, direction );

	if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
		position += direction * deltaTime * speed;
	}
	if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
		position -= direction * deltaTime * speed;
	}
	if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
		position += right * deltaTime * speed;
	}
	if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
		position -= right * deltaTime * speed;
	}
	if(glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS){
		position += up * deltaTime * speed;
	}
	if(glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS){
		position -= up * deltaTime * speed;
	}

//	float FoV = initialFoV - 5 * glfwGetMouseWheel();
	float FoV = initialFoV;

	Projection = glm::perspective(glm::radians(FoV), (float)(width / height), 0.1f, 100.0f);

	View = glm::lookAt(
		position,
		position+direction,
		up
	);

	lastTime = currentTime;
}

glm::mat4 getProjectionMatrix(){
	return Projection;
}
glm::mat4 getViewMatrix(){
	return View;
};
