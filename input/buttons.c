#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<fcntl.h>// For open function
#include<linux/joystick.h>  /* Defines all controller functions and 'struct js_event' */

/* struct js_event{
	__u32 time; // Event timestamp in ms
	__s16 value; // Value
	__u8 type; // Event type (button,axis,init)
	__u8 number; // Axis/button number
}

*/

/* Controller mapping Logitech F310 (X-input mode)
Buttons:
A-0  B-1
X-2  Y-3
LB-4  RB-5
BACK-6  START-7
HOME/LOGO-8
L3-9  R3-10

Axis (Max Val = 32767):
LX-0  RX-3
LY-1  RY-4
LT-2  RT-5
PAD X-6
PAD Y-7

*/

int main(){
	printf("LEL\n");

	int fd = open("/dev/input/js1",O_RDONLY);
	if(fd == -1){
		fprintf(stderr,"No controller to open\n");
		exit(1);
	}

	struct js_event e;
	read(fd,&e,sizeof(e));// Read block device
	e.type &= ~JS_EVENT_INIT;// Turn off bit to make no distinction between synthetic and real event
	
	while(read(fd,&e,sizeof(e)) > 0){
		if(e.type == JS_EVENT_BUTTON){
		  if(e.value) printf("Button %d: Pressed\n",e.number);
		  else printf("Button %d: Released\n",e.number);

		  if(e.number == 5 && e.value == 1) break;
		} else if(e.type == JS_EVENT_AXIS){
			printf("Axis %d: %d\n",e.number,e.value);
		}
	}

	close(fd);
}
