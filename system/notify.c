/*/
Monitor desired files using the inotify API and/or the fanotify API (if access control desired).
/*/

#include<fcntl.h>
#include<poll.h>
#include<stdio.h>
#include<string.h>
#include<sys/inotify.h>
#include<unistd.h>


int main(int argc, char *argv[]){
	const char *watch_file = "/";
	int res;

	if(geteuid() == 0){
		// TODO: Attempt to use fanotify instead (allows access control, and more options when files are manipulated)
	}

	// Setup inotify
	int inot = inotify_init();
	if(inot < 0){
		perror("inotify_init failed");
		return -1;
	}

	// Add file to watch
	res = inotify_add_watch(inot, watch_file, IN_MASK_CREATE | IN_ALL_EVENTS);
	if(res < 0){
		perror("Failed to add watch");
		return -1;
	}

	// Setup inot for poll
	struct pollfd inot_poll = {
		.fd = inot,
		.events = POLLIN,
	};

	// Loop and poll events (infinite block time)
	const size_t buffer_size = 8192;
	char buffer[buffer_size];
	ssize_t nout;
	struct inotify_event *ev = (struct inotify_event *) buffer;
	char *bp;
	while((res = poll(&inot_poll, 1, -1)) >= 0){
		// Read notify event(s)
		nout = read(inot, buffer, buffer_size);
		if(nout < 0){
			break;
		}

		if(inot_poll.revents & POLLIN){
			for(bp = buffer;(bp - buffer) < nout;bp += sizeof(struct inotify_event) + ev->len){
				ev = (struct inotify_event *) bp;

				fprintf(stdout, "Event(s) on %d '%.*s': %s %s %s %s %s %s %s %s %s %s\n",
					ev->wd,
					ev->len,
					ev->name,
					(ev->mask & IN_ISDIR) ? "ISDIR" : "",
					(ev->mask & IN_CREATE) ? "CREATED" : "",
					(ev->mask & IN_ACCESS) ? "ACCESSED" : "",
					(ev->mask & IN_ATTRIB) ? "ATTRIB" : "",
					(ev->mask & IN_OPEN) ? "OPENED" : "",
					(ev->mask & IN_MODIFY) ? "MODIFIED" : "",
					(ev->mask & IN_CLOSE) ? "CLOSED" : "",
					(ev->mask & (IN_MOVE | IN_MOVE_SELF)) ? "MOVED" : "",
					(ev->mask & (IN_DELETE_SELF | IN_DELETE)) ? "DELETED" : "",
					(ev->mask & IN_IGNORED) ? "IGNORED" : ""
				);
			}
		}
	}

	/* Cleanup
	*/
	close(inot);

	return 0;
}
