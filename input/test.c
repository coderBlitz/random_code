#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<fcntl.h>
#include<string.h>
#include<errno.h>
#include<libevdev-1.0/libevdev/libevdev.h>

int main(){
struct libevdev *dev = NULL;
	int fd;
	int rc = 1;
	fd = open("/dev/input/js1", O_RDONLY|O_NONBLOCK);// 0 is keyboard, 1/2 is mouse, 10 is trackpad
	rc = libevdev_new_from_fd(fd, &dev);
	if (rc < 0) {
        fprintf(stderr, "Failed to init libevdev (%s)\n", strerror(-rc));
	        exit(1);
	}
	printf("Input device name: \"%s\"\n", libevdev_get_name(dev));
	printf("Input device ID: bus %#x vendor %#x product %#x\n",
       libevdev_get_id_bustype(dev),
       libevdev_get_id_vendor(dev),
       libevdev_get_id_product(dev));
/*	if (!libevdev_has_event_type(dev, EV_REL) ||
    !libevdev_has_event_code(dev, EV_KEY, BTN_LEFT)) {
        printf("This device does not look like a mouse\n");
	        exit(1);// Allows codes from not usb mouse
	} */
	do {
        struct input_event ev;
	        rc = libevdev_next_event(dev, LIBEVDEV_READ_FLAG_NORMAL, &ev);
	        if (rc == 0)
                printf("Event: %s %s %d\n",
                       libevdev_event_type_get_name(ev.type),
                       libevdev_event_code_get_name(ev.type, ev.code),
                       ev.value);
	} while (rc == 1 || rc == 0 || rc == -EAGAIN);
	}
