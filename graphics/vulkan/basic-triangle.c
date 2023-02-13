/*/ The file which follows along the "Vulkan tutorial" PDF, for the simple triangle
/*/

#define GLFW_INCLUDE_NONE // Don't include OpenGL stuff
#define GLFW_INCLUDE_VULKAN
#include<GLFW/glfw3.h>
#include<error.h>
#include<stdbool.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<vulkan/vulkan.h>


VkPhysicalDevice pickDevice(const VkPhysicalDevice *const candidates, const unsigned N){
	VkPhysicalDevice ret = VK_NULL_HANDLE;

	VkPhysicalDeviceProperties vps;
	VkPhysicalDeviceFeatures vfs;
	VkQueueFamilyProperties* queue_families = VK_NULL_HANDLE;

	unsigned family_count = 0;
	for(unsigned i = 0;i < N;i++){
		vkGetPhysicalDeviceProperties(candidates[i], &vps);
		vkGetPhysicalDeviceFeatures(candidates[i], &vfs);

		vkGetPhysicalDeviceQueueFamilyProperties(candidates[i], &family_count, NULL);
		queue_families = malloc(family_count * sizeof(*queue_families));
		if(queue_families == NULL) { return VK_NULL_HANDLE; }
		vkGetPhysicalDeviceQueueFamilyProperties(candidates[i], &family_count, queue_families);

		fprintf(stdout, "Candidate %u:\n", i);
		fprintf(stdout, "\tapi = 0x%08X\n\tdriver = 0x%08X\n\tvendor = 0x%08X\n\tdevice = 0x%08X\n",
			vps.apiVersion,
			vps.driverVersion,
			vps.vendorID,
			vps.deviceID
		);
		fprintf(stdout, "\tname = %.*s\n", VK_MAX_PHYSICAL_DEVICE_NAME_SIZE, vps.deviceName);
		fprintf(stdout, "\tQ families = %u\n", family_count);

		for(unsigned j = 0;j < family_count;j++){
			fprintf(stdout, "\t\tQ family %u contains %u Q's supporting 0x%08X\n",
				j,
				queue_families[j].queueCount,
				queue_families[j].queueFlags
			);
		}

		// Select intel device always (for now)
		if(strstr(vps.deviceName, "Intel") != NULL){
			ret = candidates[i];
		}
	}

	return ret;
}

int main(int argc, char *argv[]){
	/*/ Setup window
	/*/
	const unsigned HEIGHT = 500;
	const unsigned WIDTH = 500;
	const char *title = "Vulkan - Basic triangle";

	fprintf(stdout, "Initializing GLFW..\n");
	if(!glfwInit()){
		fprintf(stderr, "GLFW init failed.\n");
		return -1;
	}

	fprintf(stdout, "Initializing window..\n");
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	//glfwWindowHint(GLFW_SAMPLES, 4);
	GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, title, NULL, NULL);

	/*/ Prep & Vulkan stuff
	/*/
	unsigned ext_count = 0;
	vkEnumerateInstanceExtensionProperties(NULL, &ext_count, NULL);
	fprintf(stdout, "%u extensions supported.\n", ext_count);

	VkApplicationInfo app_info = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "Basic Triangle",
		.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		.pEngineName = "No engi",
		.engineVersion = VK_MAKE_VERSION(1, 0, 0),
		.apiVersion = VK_API_VERSION_1_3
	};

	// GLFW fetch thing
	const char** glfwExts = glfwGetRequiredInstanceExtensions(&ext_count);

	const char *active_layers[] = {"VK_LAYER_KHRONOS_validation"};
	VkInstanceCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app_info,
		.enabledExtensionCount = ext_count,
		.ppEnabledExtensionNames = glfwExts,
		.enabledLayerCount = 1,
		.ppEnabledLayerNames = active_layers
	};

	// Vulkan init
	fprintf(stdout, "Creating vulkan instance..\n");
	VkInstance instance;
	if(vkCreateInstance(&create_info, NULL, &instance) != VK_SUCCESS){
		fprintf(stderr, "Failed to create vulkan instance\n");

		return -1;
	}

	/*/ Vulkan device stuff
	/*/
	fprintf(stdout, "Querying GPUs..\n");
	VkPhysicalDevice vk_dev = VK_NULL_HANDLE;

	// Count all physical devices
	unsigned devCount = 0;
	vkEnumeratePhysicalDevices(instance, &devCount, NULL);
	if(devCount == 0){
		fprintf(stderr, "No supported vulkan devices!\n");

		return -1;
	}

	// Fetch all physical devices
	VkPhysicalDevice *vk_devs = malloc(devCount * sizeof(*vk_devs));
	if(vk_devs == NULL){
		perror("Malloc failed");
		return -1;
	}
	vkEnumeratePhysicalDevices(instance, &devCount, vk_devs);

	// Select which device to use
	vk_dev = pickDevice(vk_devs, devCount);
	if(vk_dev == VK_NULL_HANDLE){
		fprintf(stderr, "No suitable device!\n");
		return -1;
	}

	// Create logical device (TODO: Un-hardcode queueFamilyIndex)
	VkDevice device;
	float qPriority = 1.0;
	VkDeviceQueueCreateInfo qCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = 0,
		.queueCount = 1,
		.pQueuePriorities = &qPriority
	};
	VkPhysicalDeviceFeatures devFeatures = {};
	VkDeviceCreateInfo devCreateInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.pQueueCreateInfos = &qCreateInfo,
		.queueCreateInfoCount = 1,
		.pEnabledFeatures = &devFeatures
	};

	if(vkCreateDevice(vk_dev, &devCreateInfo, NULL, &device) != VK_SUCCESS){
		fprintf(stderr, "Failed to create logical device!\n");
		return -1;
	}

	// Get new queue(s) from device
	VkQueue gfxQueue;
	vkGetDeviceQueue(device, 0, 0, &gfxQueue); // From device, from queue family 0, get queue 0

	// Create surface for link with GLFW
	VkSurfaceKHR surface;
	if(glfwCreateWindowSurface(instance, window, NULL, &surface) != VK_SUCCESS){
		fprintf(stderr, "Window surface creation error!\n");
		return -1;
	}

	//*/ Check queue present support
	VkBool32 presentSupport = false;
	vkGetPhysicalDeviceSurfaceSupportKHR(vk_dev, 0, surface, &presentSupport); // dev, queue family idx, surf, bool res

	if(!presentSupport){
		fprintf(stderr, "Device does not have present support for surface!\n");
		return -1;
	}
	//*/

	// Get present queue (XXX: Assumes gfx and present queue are the same
	VkQueue presentQueue;
	vkGetDeviceQueue(device, 0, 0, &presentQueue);

	/*/ Main loop
	/*/
	do{
		glfwPollEvents();
		//glfwSwapBuffers();
	}while(glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window));

	/*/ Cleanup stuff
	/*/
	free(vk_devs);

	vkDestroySurfaceKHR(instance, surface, NULL);
	vkDestroyDevice(device, NULL);
	vkDestroyInstance(instance, NULL);
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
