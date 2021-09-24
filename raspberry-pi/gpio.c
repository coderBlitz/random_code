#include"gpio.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define PAGE_SIZE (4*1024)
#define BLOCK_SIZE (4*1024)


// GPIO setup macros. Always use INP_GPIO(x) before using OUT_GPIO(x) or SET_GPIO_ALT(x,y)
void GPIO_INP(int g){
	*(gpio+((g)/10)) &= ~(7<<(((g)%10)*3));
}
void GPIO_OUT(int g){
	*(gpio+((g)/10)) |= (1<<(((g)%10)*3));
}
void GPIO_SET_ALT(int g,int a){
	*(gpio+(((g)/10))) |= (((a)<=3?(a)+4:(a)==4?3:2)<<(((g)%10)*3));
}
int GPIO_GET(int g){
	return (*(gpio+13)&(1<<g)); // 0 if LOW, (1<<g) if HIGH
}

void printButton(int g)
{
	if (GPIO_GET(g)) // !=0 <-> bit is 1 <- port is HIGH=3.3V
		printf("Button pressed!\n");
	else // port is LOW=0V
		printf("Button released!\n");
}


//
// Set up a memory regions to access GPIO
//
void setup_io()
{
	/* open /dev/mem */
	if ((mem_fd = open("/dev/mem", O_RDWR|O_SYNC) ) < 0) {
			printf("can't open /dev/mem \n");
			exit(-1);
	}

	/* mmap GPIO */
	gpio_map = mmap(
			NULL,			//Any adddress in our space will do
			BLOCK_SIZE,		//Map length
			PROT_READ|PROT_WRITE,	// Enable reading & writting to mapped memory
			MAP_SHARED,		//Shared with other processes
			mem_fd,			//File to map
			GPIO_BASE		//Offset to GPIO peripheral
	);

	close(mem_fd); //No need to keep mem_fd open after mmap

	if (gpio_map == MAP_FAILED) {
			printf("mmap error %d\n", (int)gpio_map);//errno also set!
			exit(-1);
	}

	// Always use volatile pointer!
	gpio = (volatile unsigned *)gpio_map;
//	printf("gpio = %X\n", gpio);

} // setup_io