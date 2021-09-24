#ifndef GPIO_H_
#define GPIO_H_


#define BCM2708_PERI_BASE	0x3F000000
#define GPIO_BASE 		(BCM2708_PERI_BASE + 0x200000) /* GPIO controller */

int mem_fd;
void *gpio_map;

// I/O access
volatile unsigned *gpio;

// GPIO setup macros. Always use INP_GPIO(x) before using OUT_GPIO(x) or SET_GPIO_ALT(x,y)
void GPIO_INP(int);
void GPIO_OUT(int);
void GPIO_SET_ALT(int,int);
int GPIO_GET(int);

#define GPIO_SET *(gpio+7)	// sets	 bits which are 1 ignores bits which are 0
#define GPIO_CLR *(gpio+10) // clears bits which are 1 ignores bits which are 0

#define GPIO_PULL *(gpio+37) // Pull up/pull down
#define GPIO_PULLCLK0 *(gpio+38) // Pull up/pull down clock


void setup_io();
void printButton(int);

#endif